"""This module contains the validators for the Guardrails framework.

The name with which a validator is registered is the name that is used
in the `RAIL` spec to specify formatters.
"""
import ast
import logging
import os
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union

import openai
from pydantic import BaseModel, ValidationError

from guardrails.datatypes import registry as types_registry
from guardrails.document_store import DocumentStoreBase
from guardrails.utils.docs_utils import sentence_split
from guardrails.utils.reask_utils import FieldReAsk
from guardrails.utils.sql_utils import SQLDriver, create_sql_driver

try:
    import numpy as np
except ImportError:
    _HAS_NUMPY = False
else:
    _HAS_NUMPY = True


validators_registry = {}
types_to_validators = defaultdict(list)


logger = logging.getLogger(__name__)


class ValidatorError(Exception):
    """Base class for all validator errors."""


class Filter:
    pass


class Refrain:
    pass


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
        elif isinstance(item, PydanticReAsk):
            filtered_list.append(item)
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
        elif isinstance(value, PydanticReAsk):
            filtered_dict[key] = value
        elif isinstance(value, list):
            filtered_item = filter_in_list(value)
            if len(filtered_item):
                filtered_dict[key] = filtered_item
        elif isinstance(value, dict):
            filtered_dict[key] = filter_in_dict(value)
        else:
            filtered_dict[key] = value

    return filtered_dict


def register_validator(name: str, data_type: Union[str, List[str]]):
    """Register a validator for a data type."""

    def decorator(cls: type):
        """Register a validator for a data type."""
        nonlocal data_type
        if isinstance(data_type, str):
            data_type = (
                list(types_registry.keys()) if data_type == "all" else [data_type]
            )
        # Make sure that the data type string exists in the data types registry.
        for dt in data_type:
            if dt not in types_registry:
                raise ValueError(f"Data type {dt} is not registered.")

            types_to_validators[dt].append(name)

        validators_registry[name] = cls
        cls.rail_alias = name
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

    def __init__(self, document_store: DocumentStoreBase, on_fail: Optional[Callable] = None, **kwargs):
        if isinstance(on_fail, str):
            self.on_fail = getattr(self, on_fail, self.noop)
        else:
            self.on_fail = on_fail or self.noop

        # Store the kwargs for the validator.
        self._kwargs = kwargs
        self._store = document_store

        assert (
            self.rail_alias in validators_registry
        ), f"Validator {self.__class__.__name__} is not registered. "

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

        error.schema[error.key] = FieldReAsk(
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

    def exception(self, error: EventDetail) -> None:
        """Raise an exception."""
        raise ValidatorError(error.error_message)

    def fix_reask(self, error: EventDetail) -> Dict:
        """If validation fails, fix the value and reask."""
        schema = self.fix(error)

        try:
            self.validate(error.key, error.fix_value, schema)
        except EventDetail as e:
            return self.reask(e)

    def to_prompt(self, with_keywords: bool = True) -> str:
        """Convert the validator to a prompt.

        E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
        ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

        Args:
            with_keywords: Whether to include the keyword arguments in the prompt.

        Returns:
            A string representation of the validator.
        """
        if not len(self._kwargs):
            return self.rail_alias

        kwargs = self._kwargs.copy()
        for k, v in kwargs.items():
            if not isinstance(v, str):
                kwargs[k] = str(v)

        params = " ".join(list(kwargs.values()))
        if with_keywords:
            params = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        return f"{self.rail_alias}: {params}"

    def to_xml_attrib(self):
        """Convert the validator to an XML attribute."""

        if not len(self._kwargs):
            return self.rail_alias

        validator_args = []
        for arg in self.__init__.__code__.co_varnames[1:]:
            if arg not in ("on_fail", "args", "kwargs"):
                str_arg = str(self._kwargs[arg])
                str_arg = "{" + str_arg + "}" if " " in str_arg else str_arg
                validator_args.append(str_arg)

        params = " ".join(validator_args)
        return f"{self.rail_alias}: {params}"

    def __call__(self, v: Any) -> Any:
        return self.validate("dummy_key", v, {"dummy_key": v})["dummy_key"]


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


class PydanticReAsk(dict):
    pass


@register_validator(name="pydantic", data_type="pydantic")
class Pydantic(Validator):
    """Validate an object using Pydantic."""

    def __init__(
        self,
        model: Type[BaseModel],
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail)

        self.model = model

    def validate_with_correction(
        self, key: str, value: Dict, schema: Union[Dict, List]
    ) -> Dict:
        """Validate an object using Pydantic.

        For example, consider the following data for a `Person` model
        with fields `name`, `age`, and `zipcode`:
        {
            "user" : {
                "name": "John",
                "age": 30,
                "zipcode": "12345",
            }
        }
        then `key` is "user", `value` is the value of the "user" key, and
        `schema` is the entire schema.

        If this validator succeeds, then the `schema` is returned and
        looks like:
        {
            "user": Person(name="John", age=30, zipcode="12345")
        }

        If it fails, then the `schema` is returned and looks like e.g.
        {
            "user": {
                "name": "John",
                "age": 30,
                "zipcode": ReAsk(
                    incorrect_value="12345",
                    error_message="...",
                    fix_value=None,
                    path=None,
                )
            }
        }
        """
        try:
            # Run the Pydantic model on the value.
            schema[key] = self.model(**value)
        except ValidationError as e:
            # Create a copy of the value so that we can modify it
            # to insert e.g. ReAsk objects.
            new_value = deepcopy(value)
            for error in e.errors():
                assert (
                    len(error["loc"]) == 1
                ), "Pydantic validation errors should only have one location."

                field_name = error["loc"][0]
                event_detail = EventDetail(
                    key=field_name,
                    value=new_value[field_name],
                    schema=new_value,
                    error_message=error["msg"],
                    fix_value=None,
                )
                # Call the on_fail method and reassign the value.
                new_value = self.on_fail(event_detail)

            # Insert the new `value` dictionary into the schema.
            # This now contains e.g. ReAsk objects.
            schema[key] = PydanticReAsk(new_value)

        return schema


@register_validator(name="pydantic_field_validator", data_type="all")
class PydanticFieldValidator(Validator):
    def __init__(
        self,
        field_validator: Callable,
        on_fail: Optional[Callable[..., Any]] = None,
        **kwargs,
    ):
        self.field_validator = field_validator
        super().__init__(on_fail, **kwargs)

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        try:
            return self.field_validator(value)
        except Exception as e:
            raise EventDetail(
                key=key,
                value=value,
                schema=schema,
                error_message=str(e),
                fix_value=None,
            )

    def to_prompt(self, with_keywords: bool = True) -> str:
        return self.field_validator.__func__.__name__


@register_validator(name="choice", data_type="choice")
class Choice(Validator):
    """Validate that a value is one of a set of choices.

    - Name for `format` attribute: `choice`
    - Supported data types: `string`
    - Programmatic fix: Closest value within the set of choices.
    """

    def __init__(
        self,
        choices: List[str],
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail, choices=choices)

        self._choices = choices

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Validate that a value is one of a set of choices."""
        logger.debug(f"Validating {value} is in {self._choices}...")

        if value not in self._choices:
            raise EventDetail(
                key=key,
                value=value,
                schema=schema,
                error_message=f"{value} is not in {self._choices}",
                fix_value=None,
            )

        selected_choice = value
        if selected_choice not in schema:
            raise EventDetail(
                key=key,
                value=value,
                schema=schema,
                error_message=f"{schema} must contain a key called {value}",
                fix_value=None,
            )

        # Make sure that no other choice is selected.
        for choice in self._choices:
            if choice == selected_choice:
                continue
            if choice in schema:
                raise EventDetail(
                    key=key,
                    value=value,
                    schema=schema,
                    error_message=(
                        f"{schema} must not contain a key called {choice}, "
                        f"since {selected_choice} is selected"
                    ),
                    fix_value=None,
                )

        return schema


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
        super().__init__(on_fail=on_fail, min=min, max=max)

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
        super().__init__(on_fail=on_fail, choices=choices)
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

        if value.lower() != value:
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

        if value.upper() != value:
            raise EventDetail(
                key,
                value,
                schema,
                f"Value {value} is not upper case.",
                value.upper(),
            )

        return schema


@register_validator(name="length", data_type=["string", "list"])
class ValidLength(Validator):
    """Validate that the length of value is within the expected range.

    - Name for `format` attribute: `length`
    - Supported data types: `string`, `list`, `object`
    - Programmatic fix: If shorter than the minimum, pad with empty last elements.
        If longer than the maximum, truncate.
    """

    def __init__(
        self, min: int = None, max: int = None, on_fail: Optional[Callable] = None
    ):
        super().__init__(on_fail=on_fail, min=min, max=max)
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
            if isinstance(value, str):
                last_val = value[-1]
            else:
                last_val = [value[-1]]

            corrected_value = value + last_val * (self._min - len(value))
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
                value[: self._max],
            )

        return schema


@register_validator(name="two-words", data_type="string")
class TwoWords(Validator):
    """Validate that a value is two words.

    - Name for `format` attribute: `two-words`
    - Supported data types: `string`
    - Programmatic fix: Pick the first two words.
    """

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        logger.debug(f"Validating {value} is two words...")

        if len(value.split()) != 2:
            raise EventDetail(
                key,
                value,
                schema,
                "must be exactly two words",
                " ".join(value.split()[:2]),
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

        if len(value.splitlines()) > 1:
            raise EventDetail(
                key,
                value,
                schema,
                f"Value {value} is not a single line.",
                value.splitlines()[0],
            )

        return schema


@register_validator(name="valid-url", data_type=["string", "url"])
class ValidURL(Validator):
    """Validate that a value is a valid URL.

    - Name for `format` attribute: `valid-url`
    - Supported data types: `string`, `url`
    - Programmatic fix: None
    """

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        logger.debug(f"Validating {value} is a valid URL...")

        from urllib.parse import urlparse

        # Check that the URL is valid
        try:
            result = urlparse(value)
            # Check that the URL has a scheme and network location
            if not result.scheme or not result.netloc:
                raise EventDetail(
                    key,
                    value,
                    schema,
                    f"URL {value} is not valid.",
                    None,
                )
        except ValueError:
            raise EventDetail(
                key,
                value,
                schema,
                f"URL {value} is not valid.",
                None,
            )

        return schema


@register_validator(name="is-reachable", data_type=["string", "url"])
class EndpointIsReachable(Validator):
    """Validate that a value is a reachable URL.

    - Name for `format` attribute: `is-reachable`
    - Supported data types: `string`, `url`
    - Programmatic fix: None
    """

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        logger.debug(f"Validating {value} is a valid URL...")

        import requests

        # Check that the URL exists and can be reached
        try:
            response = requests.get(value)
            if response.status_code != 200:
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
        except requests.exceptions.InvalidSchema:
            raise EventDetail(
                key,
                value,
                schema,
                f"URL {value} does not specify a valid connection adapter",
                None,
            )
        except requests.exceptions.MissingSchema:
            raise EventDetail(
                key,
                value,
                schema,
                f"URL {value} does not contain a http schema",
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

    def __init__(
        self,
        conn: Optional[str] = None,
        schema_file: Optional[str] = None,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail)
        self._driver: SQLDriver = create_sql_driver(schema_file=schema_file, conn=conn)

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        errors = self._driver.validate_sql(value)
        if len(errors) > 0:
            raise EventDetail(
                key,
                value,
                schema,
                ". ".join(errors),
                None,
            )

        return schema


@register_validator(name="sql-column-presence", data_type="sql")
class SqlColumnPresence(Validator):
    """Validate that all columns in the SQL query are present in the schema.

    - Name for `format` attribute: `sql-column-presence`
    - Supported data types: `string`
    """

    def __init__(self, cols: List[str], on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail, cols=cols)
        self._cols = set(cols)

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        from sqlglot import exp, parse

        expressions = parse(value)
        cols = set()
        for expression in expressions:
            for col in expression.find_all(exp.Column):
                cols.add(col.alias_or_name)

        diff = cols.difference(self._cols)
        if len(diff) > 0:
            raise EventDetail(
                key,
                value,
                schema,
                f"Columns [{', '.join(diff)}] not in [{', '.join(self._cols)}]",
                None,
            )

        return schema


@register_validator(name="exclude-sql-predicates", data_type="sql")
class ExcludeSqlPredicates(Validator):
    """Validate that the SQL query does not contain certain predicates.

    - Name for `format` attribute: `exclude-sql-predicates`
    - Supported data types: `sql`
    """

    def __init__(self, predicates: List[str], on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail, predicates=predicates)
        self._predicates = set(predicates)

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        from sqlglot import exp, parse

        expressions = parse(value)
        for expression in expressions:
            if expression is None:
                continue
            for pred in self._predicates:
                try:
                    getattr(exp, pred)
                except AttributeError:
                    raise ValueError(f"Predicate {pred} does not exist")
                if len(list(expression.find_all(getattr(exp, pred)))):
                    raise EventDetail(
                        key,
                        value,
                        schema,
                        f"SQL query contains predicate {pred}",
                        "",
                    )

        return schema


@register_validator(name="similar-to-document", data_type="string")
class SimilarToDocument(Validator):
    """Validate that a value is similar to the document.

    This validator checks if the value is similar to the document by checking
    the cosine similarity between the value and the document, using an
    embedding.

    - Name for `format` attribute: `similar-to-document`
    - Supported data types: `string`
    - Programmatic fix: None
    """

    def __init__(
        self,
        document_store: DocumentStoreBase,
        document: str,
        threshold: float = 0.7,
        model: str = "text-embedding-ada-002",
        on_fail: Optional[Callable] = None, 
    ):
        super().__init__(document_store, on_fail=on_fail)
        if not _HAS_NUMPY:
            raise ImportError(
                f"The {self.__class__.__name__} validator requires the numpy package.\n"
                "`pip install numpy` to install it."
            )

        self._document = document
        self._model = model
        self._threshold = float(threshold)
        self.document_store = document_store

    @staticmethod
    def cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
        """Calculate the cosine similarity between two vectors.

        Args:
            a: The first vector.
            b: The second vector.

        Returns:
            float: The cosine similarity between the two vectors.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        logger.debug(f"Validating {value} is similar to document...")

        document_embedding = self.document_store.add_text(self._document, {})['embeddings']
        value_embedding = self.document_store.add_text(value, schema)['embeddings']

        similarity = SimilarToDocument.cosine_similarity(
            document_embedding,
            value_embedding,
        )
        if similarity < self._threshold:
            raise EventDetail(
                key,
                value,
                schema,
                f"Value {value} is not similar enough to document {self._document}.",
                None,
            )

        return schema

    def to_prompt(self, with_keywords: bool = True) -> str:
        return ""


@register_validator(name="is-profanity-free", data_type="string")
class IsProfanityFree(Validator):
    """Validate that a translated text does not contain profanity language.

    This validator uses the `alt-profanity-check` package to check if a string
    contains profanity language.

    - Name for `format` attribute: `is-profanity-free`
    - Supported data types: `string`
    - Programmatic fix: ""
    """

    def validate(self, key, value, schema) -> Dict:
        try:
            from profanity_check import predict
        except ImportError:
            raise ImportError(
                "`is-profanity-free` validator requires the `alt-profanity-check`"
                "package. Please install it with `pip install profanity-check`."
            )

        prediction = predict([value])
        if prediction[0] == 1:
            raise EventDetail(
                key,
                value,
                schema,
                f"{value} contains profanity. Please return a profanity-free output.",
                "",
            )
        return schema


@register_validator(name="is-high-quality-translation", data_type="string")
class IsHighQualityTranslation(Validator):
    """Using inpiredco.critique to check if a translation is high quality.

    - Name for `format` attribute: `is-high-quality-translation`
    - Supported data types: `string`
    - Programmatic fix: ""
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from inspiredco.critique import Critique

            self.critique = Critique(api_key=os.environ["INSPIREDCO_API_KEY"])

        except ImportError:
            raise ImportError(
                "`is-high-quality-translation` validator requires the `inspiredco`"
                "package. Please install it with `pip install inspiredco`."
            )

    def validate(self, key, value, schema) -> Dict:
        prediction = self.critique.evaluate(
            metric="comet",
            config={"model": "unbabel_comet/wmt21-comet-qe-da"},
            dataset=[{"source": key, "target": value}],
        )
        quality = prediction["examples"][0]["value"]
        if quality < -0.1:
            raise EventDetail(
                key,
                value,
                schema,
                f"{value} is a low quality translation."
                "Please return a higher quality output.",
                "",
            )
        return schema


@register_validator(name="ends-with", data_type="list")
class EndsWith(Validator):
    """Validate that a list ends with a given value.

    - Name for `format` attribute: `ends-with`
    - Supported data types: `list`
    - Programmatic fix: Append the given value to the list.
    """

    def __init__(self, end: str, on_fail: str = "fix"):
        super().__init__(on_fail=on_fail, end=end)
        self._end = end

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        logger.debug(f"Validating {value} ends with {self._end}...")

        if not value[-1] == self._end:
            raise EventDetail(
                key,
                value,
                schema,
                f"{value} must end with {self._end}",
                value + [self._end],
            )

        return schema


@register_validator(name="extracted-summary-sentences-match", data_type="string")
class ExtractedSummarySentencesMatch(Validator):
    """Validate that the extracted summary sentences match the original text by
    performing a cosine similarity in the embedding space."""

    def __init__(
        self,
        documents_dir: str,
        document_store: DocumentStoreBase,
        threshold: float = 0.7,
        embedding_model: Optional["EmbeddingBase"] = None,  # noqa: F821
        vector_db: Optional["VectorDBBase"] = None,  # noqa: F821
        
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(on_fail, **kwargs)
        # TODO(shreya): Pass embedding_model, vector_db, document_store from spec

        if document_store is None:
            from guardrails.document_store import EphemeralDocumentStore

            if vector_db is None:
                from guardrails.vectordb import Faiss

                if embedding_model is None:
                    from guardrails.embedding import OpenAIEmbedding

                    embedding_model = OpenAIEmbedding()

                vector_db = Faiss.new_flat_ip_index(
                    embedding_model.output_dim, embedder=embedding_model
                )
            self.store = EphemeralDocumentStore(vector_db)
        else:
            self.store = document_store

        for doc_path in os.listdir(documents_dir):
            with open(os.path.join(documents_dir, doc_path)) as f:
                doc = f.read()
                self.store.add_text(
                    doc, {"path": os.path.join(documents_dir, doc_path)}
                )

        self._threshold = float(threshold)

    def validate(self, key, value, schema) -> Dict:
        # Split the value into sentences.
        sentences = re.split(r"(?<=[.!?]) +", value)

        # Check if any of the sentences in the value match any of the sentences
        # in the documents.
        unverified = []
        verified = []
        citations = []
        for sentence in sentences:
            page = self.store.search_with_threshold(sentence, self._threshold)
            if not page:
                unverified.append(sentence)
            else:
                citation_count = len(citations) + 1
                verified.append(sentence + f" [{citation_count}] ")
                citations.append(f"\n[{citation_count}] {page[0].metadata['path']}")

        fixed_summary = " ".join(verified) + "\n\n" + "".join(citations)

        if unverified:
            unverified_sentences = "\n".join(unverified)
            raise EventDetail(
                key,
                value,
                schema,
                (
                    f"The summary \nSummary: {value}\n has sentences\n"
                    f"{unverified_sentences}\n that are not similar to any document."
                ),
                fixed_summary,
            )

        schema[key] = fixed_summary
        return schema

    def to_prompt(self, with_keywords: bool = True) -> str:
        return ""


@register_validator(name="reading-time", data_type="string")
class ReadingTime(Validator):
    """Validate that the a string can be read in less than a certain amount of
    time."""

    def __init__(self, reading_time: int, on_fail: str = "fix"):
        super().__init__(on_fail=on_fail, max_time=reading_time)
        self._max_time = reading_time

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        logger.debug(
            f"Validating {value} can be read in less than {self._max_time} seconds..."
        )

        # Estimate the reading time of the string
        reading_time = len(value.split()) / 200 * 60
        logger.debug(f"Estimated reading time {reading_time} seconds...")

        if abs(reading_time - self._max_time) > 1:
            logger.error(f"{value} took {reading_time} to read")
            raise EventDetail(
                key,
                value,
                schema,
                f"String should be readable within {self._max_time} minutes.",
                value,
            )

        return schema


@register_validator(name="extractive-summary", data_type="string")
class ExtractiveSummary(Validator):
    """Validate that a string is a valid extractive summary of a given
    document.

    This validator does a fuzzy match between the sentences in the
    summary and the sentences in the document. Each sentence in the
    summary must be similar to at least one sentence in the document.
    After the validation, the summary is updated to include the
    sentences from the document that were matched, and the citations for
    those sentences are added to the end of the summary.
    """

    def __init__(
        self,
        documents_dir: str,
        threshold: int = 85,
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(on_fail, **kwargs)

        self.threshold = threshold

        # Load documents
        self._document_store = {}
        for doc_path in os.listdir(documents_dir):
            with open(os.path.join(documents_dir, doc_path)) as f:
                doc = f.read()
            self._document_store[doc_path] = sentence_split(doc)

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Make sure each sentence was precisely copied from the document."""

        try:
            from thefuzz import fuzz
        except ImportError:
            raise ImportError(
                "`thefuzz` library is required for `extractive-summary` validator. "
                "Please install it with `pip install thefuzz`."
            )

        # Split the value into sentences.
        sentences = sentence_split(value)

        # Check if any of the sentences in the value match any of the sentences
        # # in the documents.
        unverified = []
        verified = []
        citations = []

        for sentence in sentences:
            highest_ratio = 0
            highest_ratio_doc = None

            # Check fuzzy match against all sentences in all documents
            for doc_path, doc_sentences in self._document_store.items():
                for doc_sentence in doc_sentences:
                    ratio = fuzz.ratio(sentence, doc_sentence)
                    if ratio > highest_ratio:
                        highest_ratio = ratio
                        highest_ratio_doc = doc_path

            if highest_ratio < self.threshold:
                unverified.append(sentence)
            else:
                citation_count = len(citations) + 1
                verified.append(f"{sentence} [{citation_count}]")
                citations.append(f"[{citation_count}] {highest_ratio_doc}\n")

        verified_sentences = " ".join(verified) + "\n\n" + "".join(citations)

        if len(unverified):
            unverified_sentences = "\n".join(
                "- " + s for i, s in enumerate(sentences) if i in unverified
            )
            raise EventDetail(
                key,
                value,
                schema,
                (
                    f"The summary \nSummary: {value}\n has sentences\n"
                    f"{unverified_sentences}\n that are not similar to any document."
                ),
                verified_sentences,
            )

        schema[key] = verified_sentences

        return schema


@register_validator(name="remove-redundant-sentences", data_type="string")
class RemoveRedundantSentences(Validator):
    """Remove redundant sentences from a string.

    This validator removes sentences from a string that are similar to
    other sentences in the string. This is useful for removing
    repetitive sentences from a string.
    """

    def __init__(
        self, threshold: int = 70, on_fail: Optional[Callable] = None, **kwargs
    ):
        super().__init__(on_fail, **kwargs)
        self.threshold = threshold

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        """Remove redundant sentences from a string."""

        try:
            from thefuzz import fuzz
        except ImportError:
            raise ImportError(
                "`thefuzz` library is required for `remove-redundant-sentences` "
                "validator. Please install it with `pip install thefuzz`."
            )

        # Split the value into sentences.
        sentences = sentence_split(value)
        filtered_sentences = []
        redundant_sentences = []

        sentence = sentences[0]
        other_sentences = sentences[1:]
        while len(other_sentences):
            # Check fuzzy match against all other sentences
            filtered_sentences.append(sentence)
            unique_sentences = []
            for other_sentence in other_sentences:
                ratio = fuzz.ratio(sentence, other_sentence)
                if ratio > self.threshold:
                    redundant_sentences.append(other_sentence)
                else:
                    unique_sentences.append(other_sentence)
            if len(unique_sentences) == 0:
                break
            sentence = unique_sentences[0]
            other_sentences = unique_sentences[1:]

        filtered_summary = " ".join(filtered_sentences)

        if len(redundant_sentences):
            redundant_sentences = "\n".join(redundant_sentences)
            raise EventDetail(
                key,
                value,
                schema,
                (
                    f"The summary \nSummary: {value}\n has sentences\n"
                    f"{redundant_sentences}\n that are similar to other sentences."
                ),
                filtered_summary,
            )

        return schema


@register_validator(name="saliency-check", data_type="string")
class SaliencyCheck(Validator):
    """Check that the summary covers the list of topics present in the
    document."""

    def __init__(
        self,
        docs_dir: str,
        llm_callable: Callable = None,
        on_fail: Optional[Callable] = None,
        threshold: int = 0.25,
        **kwargs,
    ):
        """Initialize the SalienceCheck validator.

        Args:
            docs_dir: Path to the directory containing the documents.
            on_fail: Function to call when validation fails.
            threshold: Threshold for overlap between topics in document and summary.
        """

        super().__init__(on_fail, **kwargs)

        self.llm_callable = (
            llm_callable if llm_callable else openai.ChatCompletion.create
        )

        self.threshold = threshold

        # Load documents
        self._document_store = {}
        for doc_path in os.listdir(docs_dir):
            with open(os.path.join(docs_dir, doc_path)) as f:
                text = f.read()
            # Precompute topics for each document
            self._document_store[doc_path] = self._get_topics(text)

    @property
    def topics(self) -> List[str]:
        """Return a list of topics that can be used in the validator."""
        # Merge topics from all documents
        topics = set()
        for doc_topics in self._document_store.values():
            topics.update(doc_topics)
        return list(topics)

    def _get_topics(self, text: str, topics: Optional[List[str]] = None) -> List[str]:
        """Extract topics from a string."""

        from guardrails import Guard

        topics_seed = ""
        if topics is not None:
            topics_seed = (
                "Here's a seed list of topics, select topics from this list"
                " if they are covered in the doc:\n\n" + ", ".join(topics)
            )

        spec = f"""
<rail version="0.1">
<output>
    <list name="topics">
        <string name="topic" description="few words describing the topic in text"/>
    </list>
</output>

<prompt>
Extract a list of topics from the following text:

{text}

{topics_seed}

Return the output as a JSON with a single key "topics" containing a list of topics.

Make sure that topics are relevant to text, and topics are not too specific or general.
</prompt>
</rail>
    """

        guard = Guard.from_rail_string(spec)
        _, validated_output = guard(llm_api=self.llm_callable)
        return validated_output["topics"]

    def validate(self, key: str, value: Any, schema: Union[Dict, List]) -> Dict:
        topics_in_summary = self._get_topics(value, topics=self.topics)

        # Compute overlap between topics in document and summary
        intersection = set(topics_in_summary).intersection(set(self.topics))
        overlap = len(intersection) / len(self.topics)

        if overlap < self.threshold:
            raise EventDetail(
                key,
                value,
                schema,
                (
                    f"The summary \nSummary: {value}\n does not cover these topics:\n"
                    f"{set(self.topics).difference(intersection)}"
                ),
                "",
            )

        return schema


@register_validator(name="qa-relevance-llm-eval", data_type="string")
class QARelevanceLLMEval(Validator):
    def __init__(
        self,
        llm_callable: Callable = None,
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(on_fail, **kwargs)
        self.llm_callable = (
            llm_callable if llm_callable else openai.ChatCompletion.create
        )

    def selfeval(self, question: str, answer: str):
        from guardrails import Guard

        spec = """
<rail version="0.1">
<output>
    <bool name="relevant" />
</output>

<prompt>
Is the answer below relevant to the question asked?
Question: {question}
Answer: {answer}

Relevant (as a JSON with a single boolean key, "relevant"):\
</prompt>
</rail>
    """.format(
            question=question,
            answer=answer,
        )
        guard = Guard.from_rail_string(spec)

        return guard(
            self.llm_callable,
            max_tokens=10,
            temperature=0.1,
        )[1]

    def validate(self, key, value, schema) -> Dict:
        assert "question" in schema, "The schema must contain a `question` key."

        relevant = self.selfeval(schema["question"], value)["relevant"]
        if relevant:
            return schema

        fixed_answer = "No relevant answer found."
        raise EventDetail(
            key,
            value,
            schema,
            f"The answer {value} is not relevant to the question {schema['question']}.",
            fixed_answer,
        )

    def to_prompt(self, with_keywords: bool = True) -> str:
        return ""
