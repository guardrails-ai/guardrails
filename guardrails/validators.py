"""This module contains the validators for the Guardrails framework.

The name with which a validator is registered is the name that is used
in the `RAIL` spec to specify formatters.
"""

import ast
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from guardrails.datatypes import registry as types_registry
from guardrails.utils.reask_utils import ReAsk

validators_registry = {}
types_to_validators = defaultdict(list)


logger = logging.getLogger(__name__)


class Filter:
    pass


class Refrain:
    pass


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
        """Reask disambiguates the validation failure into a helpful error
        message."""

        error.schema[error.key] = ReAsk(
            incorrect_value=error.value,
            error_message=error.error_message,
            fix_value=error.fix_value,
        )
        return error.schema

    def filter(self, error: EventDetail) -> Dict:
        """If validation fails, filter the offending key from the schema."""

        logger.debug(f"Filtering {error.key} from schema...")

        error.schema[error.key] = Filter()

        return error.schema

    def refrain(self, error: EventDetail) -> Optional[Dict]:
        """If validation fails, refrain from answering."""

        logger.debug(f"Refusing to answer {error.key}...")

        error.schema[error.key] = Refrain()
        return error.schema

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
    """Validate that a value is within a range.

    - Name for `format` attribute: `valid-range`
    - Supported data types: `integer`, `float`, `percentage`
    - Programmatic fix: Closest value within the range.
    """

    def __init__(
        self, min: int = None, max: int = None, on_fail: Optional[Callable] = None
    ):
        super().__init__(on_fail=on_fail)

        self._min = min
        self._max = max

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Validate that a value is within a range."""

        logger.debug(f"Validating {value} is in range {self._min} - {self._max}...")

        val_type = type(value)

        if self._min is not None and value < val_type(self._min):
            raise EventDetail(
                key,
                value,
                schema,
                f"Value {value} is less than {self._min}.",
                self._min,
            )

        if self._max is not None and value > val_type(self._max):
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
    """Validate that a value is within the acceptable choices.

    - Name for `format` attribute: `valid-choices`
    - Supported data types: `all`
    - Programmatic fix: None.
    """

    def __init__(self, choices: List[Any], on_fail: Optional[Callable] = None):
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
    """Validate that a value is lower case.

    - Name for `format` attribute: `lower-case`
    - Supported data types: `string`
    - Programmatic fix: Manually convert to lower case.
    """

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
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
    """Validate that a value is upper case.

    - Name for `format` attribute: `upper-case`
    - Supported data types: `string`
    - Programmatic fix: Manually convert to upper case.
    """

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
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
    """Validate that the length of value is within the expected range.

    - Name for `format` attribute: `length`
    - Supported data types: `string`, `list`, `object`
    - Programmatic fix: If shorter than the minimum, pad with empty last elements. If longer than the maximum, truncate.  # noqa: E501
    """

    def __init__(
        self, min: int = None, max: int = None, on_fail: Optional[Callable] = None
    ):
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
                f"Please return a longer output, "
                f"that is shorter than {self._max} characters.",
                corrected_value,
            )

        if self._max is not None and len(value) > self._max:
            logger.debug(f"Value {value} is greater than {self._max}.")
            raise EventDetail(
                key,
                value,
                schema,
                f"Value has length greater than {self._max}. "
                f"Please return a shorter output, "
                f"that is shorter than {self._max} characters.",
                value[0 : self._max],
            )

        return schema


@register_validator(name="two-words", data_type="string")
class TwoWords(Validator):
    """Validate that a value is upper case.

    - Name for `format` attribute: `two-words`
    - Supported data types: `string`
    - Programmatic fix: Pick the first two words.
    """

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
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
    """Validate that a value is a single line or sentence.

    - Name for `format` attribute: `one-line`
    - Supported data types: `string`
    - Programmatic fix: Pick the first line.
    """

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
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
    """Validate that a value is a valid URL.

    - Name for `format` attribute: `valid-url`
    - Supported data types: `string`, `url`
    - Programmatic fix: None
    """

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
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
    """Validate that there are no Python syntactic bugs in the generated code.

    This validator checks for syntax errors by running `ast.parse(code)`,
    and will raise an exception if there are any.
    Only the packages in the `python` environment are available to the code snippet.

    - Name for `format` attribute: `bug-free-python`
    - Supported data types: `pythoncode`
    - Programmatic fix: None
    """

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        logger.debug(f"Validating {value} is not a bug...")

        # The value is a Python code snippet. We need to check for syntax errors.
        try:
            ast.parse(value)
        except SyntaxError as e:
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
    """Validate that there are no SQL syntactic bugs in the generated code.

    This is a very minimal implementation that uses the Pypi `sqlvalidator` package
    to check if the SQL query is valid. You can implement a custom SQL validator
    that uses a database connection to check if the query is valid.

    - Name for `format` attribute: `bug-free-sql`
    - Supported data types: `sql`
    - Programmatic fix: None
    """

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        import sqlvalidator

        sql_query = sqlvalidator.parse(value)

        if not sql_query.is_valid():
            raise EventDetail(
                key,
                value,
                schema,
                ". ".join(sql_query.errors),
                None,
            )

        return schema


def check_refrain_in_list(schema: List) -> bool:
    """Check if a Refrain object exists in a list.

    Args:
        schema: A list that can contain lists, dicts or scalars.

    Returns:
        bool: True if a Refrain object exists in the list.
    """
    for item in schema:
        if isinstance(item, Refrain):
            return True
        elif isinstance(item, list):
            if check_refrain_in_list(item):
                return True
        elif isinstance(item, dict):
            if check_refrain_in_dict(item):
                return True

    return False


def check_refrain_in_dict(schema: Dict) -> bool:
    """Check if a Refrain object exists in a dict.

    Args:
        schema: A dict that can contain lists, dicts or scalars.

    Returns:
        True if a Refrain object exists in the dict.
    """

    for key, value in schema.items():
        if isinstance(value, Refrain):
            return True
        elif isinstance(value, list):
            if check_refrain_in_list(value):
                return True
        elif isinstance(value, dict):
            if check_refrain_in_dict(value):
                return True

    return False


def filter_in_list(schema: List) -> List:
    """Remove out all Filter objects from a list.

    Args:
        schema: A list that can contain lists, dicts or scalars.

    Returns:
        A list with all Filter objects removed.
    """

    filtered_list = []

    for item in schema:
        if isinstance(item, Filter):
            pass
        elif isinstance(item, list):
            filtered_item = filter_in_list(item)
            if len(filtered_item):
                filtered_list.append(filtered_item)
        elif isinstance(item, dict):
            filtered_dict = filter_in_dict(item)
            if len(filtered_dict):
                filtered_list.append(filtered_dict)
        else:
            filtered_list.append(item)

    return filtered_list


def filter_in_dict(schema: Dict) -> Dict:
    """Remove out all Filter objects from a dictionary.

    Args:
        schema: A dictionary that can contain lists, dicts or scalars.

    Returns:
        A dictionary with all Filter objects removed.
    """

    filtered_dict = {}

    for key, value in schema.items():
        if isinstance(value, Filter):
            pass
        elif isinstance(value, list):
            filtered_item = filter_in_list(value)
            if len(filtered_item):
                filtered_dict[key] = filtered_item
        elif isinstance(value, dict):
            filtered_dict[key] = filter_in_dict(value)
        else:
            filtered_dict[key] = value

    return filtered_dict
