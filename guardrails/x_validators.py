"""Create validators for each data type."""
from collections import defaultdict
from typing import List, Union

from guardrails.x_datatypes import registry as types_registry

validators_registry = defaultdict(list)
types_to_validators = defaultdict(list)


def register_validator(name: str, data_type: Union[str, List[str]]):
    """Register a validator for a data type."""

    def decorator(cls: type):
        """Register a validator for a data type."""

        if isinstance(data_type, str):

            if data_type == 'all':
                data_type = list(types_registry.keys())

            data_type = [data_type]

        # Make sure that the data type string exists in the data types registry.
        for dt in data_type:
            if dt not in types_registry:
                raise ValueError(f"Data type {dt} is not registered.")

            types_to_validators[dt].append(name)

        validators_registry[name].append(cls)
        return cls

    return decorator


class Validator:
    """Base class for validators."""

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


@register_validator('valid-range', ['integer', 'float', 'percentage'])
class ValidRange(Validator):
    """Validate that a value is within a range."""

    def __init__(self, min: int = None, max: int = None):
        """Initialize the validator."""

        self._min = min
        self._max = max

    def validate(self, value: Any) -> bool:
        """Validate that a value is within a range."""

        if self._min is not None and value < self._min:
            return False

        if self._max is not None and value > self._max:
            return False

        return True

    def debug(self, value: Any) -> bool:
        """Validate that a value is within a range."""
        raise NotImplementedError


@register_validator('valid-choices', 'all')
class ValidChoices(Validator):
    """Validate that a value is within a range."""

    def __init__(self, choices: List[Any]):
        """Initialize the validator."""

        self._choices = choices

    def validate(self, value: Any) -> bool:
        """Validate that a value is within a range."""

        return value in self._choices

    def debug(self, value: Any) -> bool:
        """Validate that a value is within a range."""
        raise NotImplementedError