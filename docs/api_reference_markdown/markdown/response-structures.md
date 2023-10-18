# Response Structures













* guardrails.validators.ValidationResult












data*: Dict[str, Any] | None*

#### outcome*: str*

### *class* guardrails.validators.PassResult

#### *class* ValueOverrideSentinel

#### outcome*: Literal['pass']*

#### value_override*: Any | None*

### *class* guardrails.validators.FailResult

#### error_message*: str*

#### fix_value*: Any | None*

#### outcome*: Literal['fail']*

### *class* guardrails.validators.ValidatorError

Base class for all validator errors.
all validator errors.
