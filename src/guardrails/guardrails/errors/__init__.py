class ValidationError(Exception):
    """Top level validation error.

    This is thrown from the validation engine when a Validator has
    on_fail=OnFailActions.EXCEPTION set and validation fails.

    Inherits from Exception.
    """


class UserFacingException(Exception):
    """Wraps an exception to denote it as user-facing.

    It will be unwrapped in runner.
    """

    def __init__(self, original_exception: Exception):
        super().__init__()
        self.original_exception = original_exception


__all__ = ["ValidationError", "UserFacingException"]
