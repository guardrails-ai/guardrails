"""Create validators for each data type."""
import logging
from collections import defaultdict
from typing import List, Union, Any, Optional, Callable

from guardrails.x_datatypes import registry as types_registry

validators_registry = {}
types_to_validators = defaultdict(list)

# Set up logger

logger = logging.getLogger(__name__)


def register_validator(name: str, data_type: Union[str, List[str]]):
    """Register a validator for a data type."""

    def decorator(cls: type):
        """Register a validator for a data type."""

        nonlocal data_type
        if isinstance(data_type, str):
            if data_type == 'all':
                data_type = list(types_registry.keys())
            else:
                data_type = [data_type]

        # Make sure that the data type string exists in the data types registry.
        for dt in data_type:
            if dt not in types_registry:
                raise ValueError(f"Data type {dt} is not registered.")

            types_to_validators[dt].append(name)

        validators_registry[name] = cls
        return cls

    return decorator


class Validator:
    """Base class for validators."""

    def __init__(self, on_fail: Optional[Callable] = None):
        if on_fail is not None:
            self.on_fail = on_fail
        else:
            self.on_fail = self.debug

    def validate(self, value: Any) -> bool:
        """Validate a value."""

        raise NotImplementedError

    def debug(self, value: Any) -> bool:
        """Validate a value."""

        raise NotImplementedError


# @register_validator('required', 'all')
# class Required(Validator):
#     """Validate that a value is not None."""

#     def validate(self, value: Any) -> bool:
#         """Validate that a value is not None."""

#         return value is not None


# @register_validator('description', 'all')
# class Description(Validator):
#     """Validate that a value is not None."""

#     def validate(self, value: Any) -> bool:
#         """Validate that a value is not None."""

#         return value is not None


@register_validator(name='valid-range', data_type=['integer', 'float', 'percentage'])
class ValidRange(Validator):
    """Validate that a value is within a range."""

    def __init__(self, min: int = None, max: int = None, on_fail: Optional[Callable] = None):
        """Initialize the validator."""
        super().__init__(on_fail=on_fail)

        self._min = min
        self._max = max

    def validate(self, value: Any) -> bool:
        """Validate that a value is within a range."""

        logger.debug(f"Validating {value} is in range {self._min} - {self._max}...")

        if self._min is not None and value < self._min:
            logger.debug(f"Value {value} is less than {self._min}.")
            return False

        if self._max is not None and value > self._max:
            logger.debug(f"Value {value} is greater than {self._max}.")
            return False

        return True

    def debug(self, value: Any) -> bool:
        """Validate that a value is within a range."""
        raise NotImplementedError


@register_validator(name='valid-choices', data_type='all')
class ValidChoices(Validator):
    """Validate that a value is within a range."""

    def __init__(self, choices: List[Any], on_fail: Optional[Callable] = None):
        """Initialize the validator."""
        super().__init__(on_fail=on_fail)
        self._choices = choices

    def validate(self, value: Any) -> bool:
        """Validate that a value is within a range."""

        logger.debug(f"Validating {value} is in choices {self._choices}...")

        validation_outcome = value in self._choices

        logger.debug(f"Validation outcome: {validation_outcome}")

        return validation_outcome

    def debug(self, value: Any) -> bool:
        """Validate that a value is within a range."""
        raise NotImplementedError


@register_validator(name='lower-case', data_type='string')
class LowerCase(Validator):
    """Validate that a value is lower case."""

    def validate(self, value: Any) -> bool:
        """Validate that a value is lower case."""

        logger.debug(f"Validating {value} is lower case...")

        validation_outcome = value.lower() == value

        logger.debug(f"Validation outcome: {validation_outcome}")

        return validation_outcome

    def debug(self, value: Any) -> bool:
        """Validate that a value is lower case."""
        raise NotImplementedError


@register_validator(name='upper-case', data_type='string')
class UpperCase(Validator):
    """Validate that a value is upper case."""

    def validate(self, value: Any) -> bool:
        """Validate that a value is upper case."""

        logger.debug(f"Validating {value} is upper case...")

        validation_outcome = value.upper() == value

        logger.debug(f"Validation outcome: {validation_outcome}")

        return validation_outcome

    def debug(self, value: Any) -> bool:
        """Validate that a value is upper case."""
        raise NotImplementedError


@register_validator(name='length', data_type=['string', 'list', 'object'])
class ValidLength(Validator):
    """Validate that the length of value is within the expected range."""

    def __init__(
            self,
            min: int = None,
            max: int = None,
            on_fail: Optional[Callable] = None
    ):
        """Initialize the validator."""
        super().__init__(on_fail=on_fail)
        self._min = int(min)
        self._max = int(max)

    def validate(self, value: Any) -> bool:
        """Validate that a value is within a range."""

        logger.debug(f"Validating {value} is in length range {self._min} - {self._max}...")

        if self._min is not None and len(value) < self._min:
            logger.debug(f"Value {value} is less than {self._min}.")
            return False

        if self._max is not None and len(value) > self._max:
            logger.debug(f"Value {value} is greater than {self._max}.")
            return False

        logger.debug(f"Value {value} is in range {self._min} - {self._max}.")
        return True

    def debug(self, value: Any) -> bool:
        """Validate that a value is within a range."""
        raise NotImplementedError


@register_validator(name='two-words', data_type='string')
class TwoWords(Validator):
    """Validate that a value is upper case."""

    def validate(self, value: Any) -> bool:
        """Validate that a value is upper case."""
        logger.debug(f"Validating {value} is two words...")

        validation_outcome = len(value.split()) == 2

        logger.debug(f"Validation outcome: {validation_outcome}")

        return validation_outcome

    def debug(self, value: Any) -> bool:
        """Validate that a value is upper case."""
        raise NotImplementedError


@register_validator(name='one-line', data_type='string')
class OneLine(Validator):
    """Validate that a value is a single line or sentence."""

    def validate(self, value: Any) -> bool:
        """Validate that a value is a single line or sentence."""
        logger.debug(f"Validating {value} is a single line...")

        validation_outcome = len(value.splitlines()) == 1

        logger.debug(f"Validation outcome: {validation_outcome}")

        return validation_outcome

    def debug(self, value: Any) -> bool:
        """Validate that a value is upper case."""
        raise NotImplementedError
