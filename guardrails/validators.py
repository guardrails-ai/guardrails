import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Union, Any, Optional, Callable, Dict

from guardrails.datatypes import registry as types_registry
from guardrails.utils.reask_utils import ReAsk

validators_registry = {}
types_to_validators = defaultdict(list)


logger = logging.getLogger(__name__)


def register_validator(name: str, data_type: Union[str, List[str]]):
    """Register a validator for a data type."""

    def decorator(cls: type):
        """Register a validator for a data type."""

        nonlocal data_type
        if isinstance(data_type, str):
            if data_type == "all":
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


@dataclass
class EventDetail(BaseException):
    """Event detail."""

    key: str
    value: Any
    schema: Dict[str, Any]
    error_message: str
    fix_value: Any


class Validator:
    """Base class for validators."""

    def __init__(self, on_fail: Optional[Callable] = None):
        if on_fail is not None:
            if isinstance(on_fail, str):
                if on_fail == "filter":
                    on_fail = self.filter
                elif on_fail == "refrain":
                    on_fail = self.refrain
                elif on_fail == "noop":
                    on_fail = self.noop
                elif on_fail == "fix":
                    on_fail = self.fix
                elif on_fail == "reask":
                    on_fail = self.reask
                else:
                    raise ValueError(f"Unknown on_fail value: {on_fail}.")
            self.on_fail = on_fail
        else:
            self.on_fail = self.fix

    def validate_with_correction(self, key, value, schema) -> Dict:
        try:
            return self.validate(key, value, schema)
        except EventDetail as e:
            logger.debug(
                f"Validator {self.__class__.__name__} failed for {key} with error {e}."
            )
            return self.on_fail(e)

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Validate a value."""

        raise NotImplementedError

    def fix(self, error: EventDetail) -> Dict:
        """Debug the incorrect value."""

        error.schema[error.key] = error.fix_value
        return error.schema

    def reask(self, error: EventDetail) -> Dict:
        """Reask disambiguates the validation failure into a helpful error message."""

        error.schema[error.key] = ReAsk(error.value, error.error_message)
        return error.schema

    def filter(self, error: EventDetail) -> Dict:
        """If validation fails, filter the offending key from the schema."""

        logger.debug(f"Filtering {error.key} from schema...")

        schema = error.schema
        key = error.key
        value = error.value

        if isinstance(schema, dict):
            schema.pop(key)
        elif isinstance(schema, list):
            schema.remove(value)

        return schema

    def refrain(self, error: EventDetail) -> Optional[Dict]:
        """If validation fails, refrain from answering."""

        logger.debug(f"Refusing to answer {error.key}...")

        return None

    def noop(self, error: EventDetail) -> Dict:
        """If validation fails, do nothing."""

        logger.debug(
            f"Validator {self.__class__.__name__} failed for {error.key}, "
            "but doing nothing..."
        )

        return error.schema


# @register_validator('required', 'all')
# class Required(Validator):
#     """Validate that a value is not None."""

#     def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> bool:
#         """Validate that a value is not None."""

#         return value is not None


# @register_validator('description', 'all')
# class Description(Validator):
#     """Validate that a value is not None."""

#     def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> bool:
#         """Validate that a value is not None."""

#         return value is not None


@register_validator(name="valid-range", data_type=["integer", "float", "percentage"])
class ValidRange(Validator):
    """Validate that a value is within a range."""

    def __init__(
        self, min: int = None, max: int = None, on_fail: Optional[Callable] = None
    ):
        """Initialize the validator."""
        super().__init__(on_fail=on_fail)

        self._min = min
        self._max = max

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Validate that a value is within a range."""

        logger.debug(f"Validating {value} is in range {self._min} - {self._max}...")

        if self._min is not None and value < self._min:
            raise EventDetail(
                key,
                value,
                schema,
                f"Value {value} is less than {self._min}.",
                self._min,
            )

        if self._max is not None and value > self._max:
            raise EventDetail(
                key,
                value,
                schema,
                f"Value {value} is greater than {self._max}.",
                self._max,
            )

        return schema


@register_validator(name="valid-choices", data_type="all")
class ValidChoices(Validator):
    """Validate that a value is within a range."""

    def __init__(self, choices: List[Any], on_fail: Optional[Callable] = None):
        """Initialize the validator."""
        super().__init__(on_fail=on_fail)
        self._choices = choices

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Validate that a value is within a range."""

        logger.debug(f"Validating {value} is in choices {self._choices}...")

        if value not in self._choices:
            raise EventDetail(
                key,
                value,
                schema,
                f"Value {value} is not in choices {self._choices}.",
                None,
            )

        return schema


@register_validator(name="lower-case", data_type="string")
class LowerCase(Validator):
    """Validate that a value is lower case."""

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Validate that a value is lower case."""

        logger.debug(f"Validating {value} is lower case...")

        if not value.lower() == value:
            raise EventDetail(
                key,
                value,
                schema,
                f"Value {value} is not lower case.",
                value.lower(),
            )

        return schema


@register_validator(name="upper-case", data_type="string")
class UpperCase(Validator):
    """Validate that a value is upper case."""

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Validate that a value is upper case."""

        logger.debug(f"Validating {value} is upper case...")

        if not value.upper() == value:
            raise EventDetail(
                key,
                value,
                schema,
                f"Value {value} is not upper case.",
                value.upper(),
            )

        return schema


@register_validator(name="length", data_type=["string", "list", "object"])
class ValidLength(Validator):
    """Validate that the length of value is within the expected range."""

    def __init__(
        self, min: int = None, max: int = None, on_fail: Optional[Callable] = None
    ):
        """
        Args:
            min: The minimum length of the value.
            max: The maximum length of the value.
        """
        super().__init__(on_fail=on_fail)
        self._min = int(min) if min is not None else None
        self._max = int(max) if max is not None else None

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Validate that a value is within a range."""

        logger.debug(
            f"Validating {value} is in length range {self._min} - {self._max}..."
        )

        if self._min is not None and len(value) < self._min:
            logger.debug(f"Value {value} is less than {self._min}.")

            # Repeat the last character to make the value the correct length.
            corrected_value = value + value[-1] * (self._min - len(value))
            raise EventDetail(
                key,
                value,
                schema,
                f"Value has length less than {self._min}. "
                f"Please return a longer output, that is shorter than {self._max} characters.",
                corrected_value,
            )

        if self._max is not None and len(value) > self._max:
            logger.debug(f"Value {value} is greater than {self._max}.")
            raise EventDetail(
                key,
                value,
                schema,
                f"Value has length greater than {self._max}. "
                f"Please return a shorter output, that is shorter than {self._max} characters.",
                value[0 : self._max],
            )

        return schema


@register_validator(name="two-words", data_type="string")
class TwoWords(Validator):
    """Validate that a value is upper case."""

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Validate that a value is upper case."""
        logger.debug(f"Validating {value} is two words...")

        if not len(value.split()) == 2:
            raise EventDetail(
                key,
                value,
                schema,
                "must be exactly two words",
                " ".join(value.split()[0:2]),
            )

        return schema


@register_validator(name="one-line", data_type="string")
class OneLine(Validator):
    """Validate that a value is a single line or sentence."""

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Validate that a value is a single line or sentence."""
        logger.debug(f"Validating {value} is a single line...")

        if not len(value.splitlines()) == 1:
            raise EventDetail(
                key,
                value,
                schema,
                f"Value {value} is not a single line.",
                value.splitlines()[0],
            )

        return schema


@register_validator(name="valid-url", data_type=["string", "url"])
class ValidUrl(Validator):
    """Validate that a value is a valid URL."""

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Validate that a value is a valid URL."""
        logger.debug(f"Validating {value} is a valid URL...")

        import requests

        # Check that the URL exists and can be reached
        try:
            response = requests.get(value)
            if not response.status_code == 200:
                raise EventDetail(
                    key,
                    value,
                    schema,
                    f"URL {value} returned status code {response.status_code}",
                    None,
                )
        except requests.exceptions.ConnectionError:
            raise EventDetail(
                key,
                value,
                schema,
                f"URL {value} could not be reached",
                None,
            )

        return schema


@register_validator(name="bug-free-python", data_type="pythoncode")
class BugFreePython(Validator):
    """Validate that a value is not a bug."""

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Validate that a value is not a bug."""
        logger.debug(f"Validating {value} is not a bug...")

        # The value represents a Python code snippet. We need to execute it and check if there are any bugs
        try:
            exec(value)
        except Exception as e:
            raise EventDetail(
                key,
                value,
                schema,
                e,
                None,
            )

        return schema


@register_validator(name="bug-free-sql", data_type="sql")
class BugFreeSQL(Validator):
    """Validate that a value is not a bug."""

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Validate that a value is not a bug."""

        import sqlvalidator

        sql_query = sqlvalidator.parse("SELECT * from table")

        if not sql_query.is_valid():
            raise EventDetail(
                key,
                value,
                schema,
                '. '.join(sql_query.errors),
                None,
            )

        return schema
