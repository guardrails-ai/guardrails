# Validation

## Validator

```python
@dataclass
class Validator()
```

Base class for validators.

#### \_\_init\_\_

```python
def __init__(on_fail: Optional[Union[Callable[[Any, FailResult], Any],
                                     OnFailAction]] = None,
             **kwargs)
```

#### validate

```python
def validate(value: Any, metadata: Dict[str, Any]) -> ValidationResult
```

Do not override this function, instead implement _validate().

External facing validate function. This function acts as a
wrapper for _validate() and is intended to apply any meta-
validation requirements, logic, or pre/post processing.

#### validate\_stream

```python
def validate_stream(chunk: Any,
                    metadata: Dict[str, Any],
                    *,
                    property_path: Optional[str] = "$",
                    context_vars: Optional[ContextVar[Dict[
                        str, ContextVar[List[str]]]]] = None,
                    context: Optional[Context] = None,
                    **kwargs) -> Optional[ValidationResult]
```

Validates a chunk emitted by an LLM. If the LLM chunk is smaller
than the validator's chunking strategy, it will be accumulated until it
reaches the desired size. In the meantime, the validator will return
None.

If the LLM chunk is larger than the validator's chunking
strategy, it will split it into validator-sized chunks and
validate each one, returning an array of validation results.

Otherwise, the validator will validate the chunk and return the
result.

#### with\_metadata

```python
def with_metadata(metadata: Dict[str, Any])
```

Assigns metadata to this validator to use during validation.

#### to\_runnable

```python
def to_runnable() -> Runnable
```

#### register\_validator

```python
def register_validator(
    name: str,
    data_type: Union[str, List[str]],
    has_guardrails_endpoint: bool = False
) -> Callable[[Union[Type[V], Callable]], Union[Type[V], Type[Validator]]]
```

Register a validator for a data type.

## ValidationResult

```python
class ValidationResult(IValidationResult, ArbitraryModel)
```

ValidationResult is the output type of Validator.validate and the
abstract base class for all validation results.

**Attributes**:

- `outcome` _str_ - The outcome of the validation. Must be one of "pass" or "fail".
- `metadata` _Optional[Dict[str, Any]]_ - The metadata associated with this
  validation result.
- `validated_chunk` _Optional[Any]_ - The value argument passed to
  validator.validate or validator.validate_stream.

## PassResult

```python
class PassResult(ValidationResult, IPassResult)
```

PassResult is the output type of Validator.validate when validation
succeeds.

**Attributes**:

- `outcome` _Literal["pass"]_ - The outcome of the validation. Must be "pass".
- `value_override` _Optional[Any]_ - The value to use as an override
  if validation passes.

## FailResult

```python
class FailResult(ValidationResult, IFailResult)
```

FailResult is the output type of Validator.validate when validation
fails.

**Attributes**:

- `outcome` _Literal["fail"]_ - The outcome of the validation. Must be "fail".
- `error_message` _str_ - The error message indicating why validation failed.
- `fix_value` _Optional[Any]_ - The auto-fix value that would be applied
  if the Validator's on_fail method is "fix".
- `error_spans` _Optional[List[ErrorSpan]]_ - Segments that caused
  validation to fail.

## ErrorSpan

```python
class ErrorSpan(IErrorSpan, ArbitraryModel)
```

ErrorSpan provide additional context for why a validation failed. They
specify the start and end index of the segment that caused the failure,
which can be useful when validating large chunks of text or validating
while streaming with different chunking methods.

**Attributes**:

- `start` _int_ - Starting index relative to the validated chunk.
- `end` _int_ - Ending index relative to the validated chunk.
- `reason` _str_ - Reason validation failed for this chunk.

## ValidatorLogs

```python
class ValidatorLogs(IValidatorLog, ArbitraryModel)
```

Logs for a single validator execution.

**Attributes**:

- `validator_name` _str_ - The class name of the validator
- `registered_name` _str_ - The snake_cased id of the validator
- `property_path` _str_ - The JSON path to the property being validated
- `value_before_validation` _Any_ - The value before validation
- `value_after_validation` _Optional[Any]_ - The value after validation;
  could be different if `value_override`s or `fix`es are applied
- `validation_result` _Optional[ValidationResult]_ - The result of the validation
- `start_time` _Optional[datetime]_ - The time the validation started
- `end_time` _Optional[datetime]_ - The time the validation ended
- `instance_id` _Optional[int]_ - The unique id of this instance of the validator

## ValidatorReference

```python
class ValidatorReference(IValidatorReference)
```

ValidatorReference is a serialized reference for constructing a
Validator.

**Attributes**:

- `id` _Optional[str]_ - The unique identifier for this Validator.
  Often the hub id; e.g. guardrails/regex_match.  Default None.
- `on` _Optional[str]_ - A reference to the property this validator should be
  applied against.  Can be a valid JSON path or a meta-property
  such as `prompt` or `output`. Default None.
- `on_fail` _Optional[str]_ - The OnFailAction to apply during validation.
  Default None.
- `args` _Optional[List[Any]]_ - Positional arguments. Default None.
- `kwargs` _Optional[Dict[str, Any]]_ - Keyword arguments. Default None.

