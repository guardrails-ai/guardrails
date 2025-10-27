# Errors

## ValidationError

```python
class ValidationError(Exception)
```

Top level validation error.

This is thrown from the validation engine when a Validator has
on_fail=OnFailActions.EXCEPTION set and validation fails.

Inherits from Exception.

