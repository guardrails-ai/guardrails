# Never actually used in any validators so the description is misleading.
# The naming is confusing so we're updating it.
class ValidatorError(Exception):
    """
    deprecated: 0.3.3
    Use :class:`ValidationError` instead.

    Base class for all validator errors.
    """


# Open to naming this something more generic like GuardrailsError or something,
# let's just decide in this PR
class ValidationError(Exception):
    """Top level validation error."""


__all__ = ["ValidatorError", "ValidationError"]
