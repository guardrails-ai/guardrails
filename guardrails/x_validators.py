"""Create validators for each data type."""
from collections import defaultdict
from typing import List, Union, Any, Optional, Callable

from guardrails.x_datatypes import registry as types_registry

validators_registry = {}
types_to_validators = defaultdict(list)


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

        if self._min is not None and value < self._min:
            return False

        if self._max is not None and value > self._max:
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

        return value in self._choices

    def debug(self, value: Any) -> bool:
        """Validate that a value is within a range."""
        raise NotImplementedError