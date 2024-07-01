class ValidationError(Exception):
    """Top level validation error."""


class UserFacingException(Exception):
    """Wraps an exception to denote it as user-facing.

    It will be unwrapped in runner.
    """

    def __init__(self, original_exception: Exception):
        super().__init__()
        self.original_exception = original_exception


__all__ = ["ValidationError", "UserFacingException"]
