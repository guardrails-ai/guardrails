import logging
import pprint
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from lxml import etree as ET

from guardrails.datatypes import DataType
from guardrails.validators import Validator, check_refrain_in_dict, filter_in_dict

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class FormatAttr:
    """Class for parsing and manipulating the `format` attribute of an element.

    The format attribute is a string that contains semi-colon separated
    validators e.g. "valid-url; is-reachable". Each validator is itself either:
    - the name of an parameter-less validator, e.g. "valid-url"
    - the name of a validator with parameters, separated by a colon with a
        comma-separated list of parameters, e.g. "is-in: 1, 2, 3"

    Parameters can either be written in plain text, or in python expressions
    enclosed in curly braces. For example, the following are all valid:
    - "is-in: 1, 2, 3"
    - "is-in: {1}, {2}, {3}"
    - "is-in: {1 + 2}, {2 + 3}, {3 + 4}"
    """

    # The format attribute string.
    format: Optional[str] = None

    # The XML element that this format attribute is associated with.
    element: Optional[ET._Element] = None

    @classmethod
    def from_element(cls, element: ET._Element) -> "FormatAttr":
        """Create a FormatAttr object from an XML element.

        Args:
            element (ET._Element): The XML element.

        Returns:
            A FormatAttr object.
        """
        return cls(element.get("format"), element)

    @property
    def tokens(self) -> List[str]:
        """Split the format attribute into tokens.

        For example, the format attribute "valid-url; is-reachable" will
        be split into ["valid-url", "is-reachable"]. The semicolon is
        used as a delimiter, but not if it is inside curly braces,
        because the format string can contain Python expressions that
        contain semicolons.
        """
        if self.format is None:
            return []
        pattern = re.compile(r";(?![^{}]*})")
        tokens = re.split(pattern, self.format)
        tokens = list(filter(None, tokens))
        return tokens

    def parse_token(self, token: str) -> Tuple[str, List[Any]]:
        """Parse a single token in the format attribute, and return the
        validator name and the list of arguments.

        Args:
            token (str): The token to parse, one of the tokens returned by
                `self.tokens`.

        Returns:
            A tuple of the validator name and the list of arguments.
        """
        validator_with_args = token.strip().split(":", 1)
        if len(validator_with_args) == 1:
            return validator_with_args[0].strip(), []

        validator, args_token = validator_with_args

        # Split using whitespace as a delimiter, but not if it is inside curly braces or
        # single quotes.
        pattern = re.compile(r"\s(?![^{}]*})|(?<!')\s(?=[^']*'$)")
        tokens = re.split(pattern, args_token)

        # Filter out empty strings if any.
        tokens = list(filter(None, tokens))

        args = []
        for t in tokens:
            # If the token is enclosed in curly braces, it is a Python expression.
            t = t.strip()
            if t[0] == "{" and t[-1] == "}":
                t = t[1:-1]
                try:
                    # Evaluate the Python expression.
                    t = eval(t)
                except (ValueError, SyntaxError, NameError) as e:
                    raise ValueError(
                        f"Python expression `{t}` is not valid, "
                        f"and raised an error: {e}."
                    )
            args.append(t)

        return validator.strip(), args

    def parse(self) -> Dict:
        """Parse the format attribute into a dictionary of validators.

        Returns:
            A dictionary of validators, where the key is the validator name, and
            the value is a list of arguments.
        """
        if self.format is None:
            return {}

        # Split the format attribute into tokens: each is a validator.
        # Then, parse each token into a validator name and a list of parameters.
        validators = {}
        for token in self.tokens:
            # Parse the token into a validator name and a list of parameters.
            validator_name, args = self.parse_token(token)
            validators[validator_name] = args

        return validators

    @property
    def validators(self) -> List[Validator]:
        """Get the list of validators from the format attribute.

        Only the validators that are registered for this element will be
        returned.
        """
        try:
            return getattr(self, "_validators")
        except AttributeError:
            raise AttributeError("Must call `get_validators` first.")

    def get_validators(self, strict: bool = False) -> List[Validator]:
        """Get the list of validators from the format attribute. Only the
        validators that are registered for this element will be returned.

        For example, if the format attribute is "valid-url; is-reachable", and
        "is-reachable" is not registered for this element, then only the ValidUrl
        validator will be returned, after instantiating it with the arguments
        specified in the format attribute (if any).

        Args:
            strict: If True, raise an error if a validator is not registered for
                this element. If False, ignore the validator and print a warning.

        Returns:
            A list of validators.
        """
        from guardrails.validators import types_to_validators, validators_registry

        _validators = []
        parsed = self.parse().items()
        for validator_name, args in parsed:
            # Check if the validator is registered for this element.
            # The validators in `format` that are not registered for this element
            # will be ignored (with an error or warning, depending on the value of
            # `strict`), and the registered validators will be returned.
            if validator_name not in types_to_validators[self.element.tag]:
                if strict:
                    raise ValueError(
                        f"Validator {validator_name} is not valid for"
                        f" element {self.element.tag}."
                    )
                else:
                    warnings.warn(
                        f"Validator {validator_name} is not valid for"
                        f" element {self.element.tag}."
                    )
                continue

            validator = validators_registry[validator_name]

            # See if the formatter has an associated on_fail method.
            on_fail = None
            on_fail_attr_name = f"on-fail-{validator_name}"
            if on_fail_attr_name in self.element.attrib:
                on_fail = self.element.attrib[on_fail_attr_name]
                # TODO(shreya): Load the on_fail method.
                # This method should be loaded from an optional script given at the
                # beginning of a rail file.

            # Create the validator.
            _validators.append(validator(*args, on_fail=on_fail))

        self._validators = _validators
        return _validators

    def to_prompt(self, with_keywords: bool = True) -> str:
        """Convert the format string to another string representation for use
        in prompting. Uses the validators' to_prompt method in order to
        construct the string to use in prompting.

        For example, the format string "valid-url; other-validator: 1.0
        {1 + 2}" will be converted to "valid-url other-validator:
        arg1=1.0 arg2=3".
        """
        if self.format is None:
            return ""
        # Use the validators' to_prompt method to convert the format string to
        # another string representation.
        return "; ".join([v.to_prompt(with_keywords) for v in self.validators])


class Schema:
    """Schema class that holds a _schema attribute."""

    def __init__(
        self,
        root: Optional[ET._Element] = None,
        schema: Optional[Dict[str, DataType]] = None,
    ) -> None:
        from guardrails.datatypes import registry as types_registry

        if schema is None:
            schema = {}

        self._schema = SimpleNamespace(**schema)
        self.root = root

        if root is not None:
            strict = False
            if "strict" in root.attrib and root.attrib["strict"] == "true":
                strict = True

            for child in root:
                if isinstance(child, ET._Comment):
                    continue
                child_name = child.attrib["name"]
                child_data = types_registry[child.tag].from_xml(child, strict=strict)
                self[child_name] = child_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({pprint.pformat(vars(self._schema))})"

    def __getitem__(self, key: str) -> DataType:
        return getattr(self._schema, key)

    def __setitem__(self, key: str, value: DataType) -> None:
        setattr(self._schema, key, value)

    def __getattr__(self, key: str) -> DataType:
        return getattr(self._schema, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self._schema, key)

    def __getstate__(self) -> Dict[str, Any]:
        return {"_schema": self._schema, "root": self.root}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._schema = state["_schema"]
        self.root = state["root"]

    def items(self) -> Dict[str, DataType]:
        return vars(self._schema).items()

    @property
    def parsed_rail(self) -> Optional[ET._Element]:
        return self.root

    def validate(
        self,
        data: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Validate a dictionary of data against the schema.

        Args:
            data: The data to validate.

        Returns:
            The validated data.
        """
        if data is None:
            return None

        if not isinstance(data, dict):
            raise TypeError(f"Argument `data` must be a dictionary, not {type(data)}.")

        validated_response = deepcopy(data)

        for field, value in validated_response.items():
            if field not in self:
                # This is an extra field that is not in the schema.
                # We remove it from the validated response.
                logger.debug(f"Field {field} not in schema.")
                continue

            validated_response = self[field].validate(
                key=field,
                value=value,
                schema=validated_response,
            )

        if check_refrain_in_dict(validated_response):
            # If the data contains a `Refain` value, we return an empty
            # dictionary.
            logger.debug("Refrain detected.")
            validated_response = {}

        # Remove all keys that have `Filter` values.
        validated_response = filter_in_dict(validated_response)

        return validated_response

    def transpile(self, method: str = "default") -> str:
        """Convert the XML schema to a string that is used for prompting a
        large language model.

        Returns:
            The prompt.
        """
        transpiler = getattr(Schema2Prompt, method)
        return transpiler(self)


class InputSchema(Schema):
    """Input schema class that holds a _schema attribute."""


class OutputSchema(Schema):
    """Output schema class that holds a _schema attribute."""


class Schema2Prompt:
    """Class that contains transpilers to go from a schema to its
    representation in a prompt.

    This is important for communicating the schema to a large language
    model, and this class will provide multiple alternatives to do so.
    """

    @staticmethod
    def remove_on_fail_attributes(element: ET._Element) -> None:
        """Recursively remove all attributes that start with 'on-fail-'."""
        for attr in list(element.attrib):
            if attr.startswith("on-fail-"):
                del element.attrib[attr]

        for child in element:
            Schema2Prompt.remove_on_fail_attributes(child)

    @staticmethod
    def remove_comments(element: ET._Element) -> None:
        """Recursively remove all comments."""
        for child in element:
            if isinstance(child, ET._Comment):
                element.remove(child)
            else:
                Schema2Prompt.remove_comments(child)

    @staticmethod
    def validator_to_prompt(schema: Schema) -> None:
        """Recursively remove all validator arguments in the `format`
        attribute."""

        def _inner(dt: DataType, el: ET._Element):
            if hasattr(el.attrib, "format"):
                el.attrib["format"] = dt.format_attr.to_prompt()

            for _, dt_child, el_child in dt:
                _inner(dt_child, el_child)

        for el_child in schema.root:
            dt_child = schema[el_child.attrib["name"]]
            _inner(dt_child, el_child)

    @classmethod
    def default(cls, schema: Schema) -> str:
        """Default transpiler.

        Converts the XML schema to a string directly after removing:
            - Comments
            - Action attributes like 'on-fail-*'

        Args:
            schema: The schema to transpile.

        Returns:
            The prompt.
        """
        # Construct another XML tree from the schema.
        # root = deepcopy(schema.root)
        schema = deepcopy(schema)

        for k, v in schema.items():
            print(k, v)

        # Remove comments.
        cls.remove_comments(schema.root)
        # Remove action attributes.
        cls.remove_on_fail_attributes(schema.root)
        # Remove validators with arguments.
        cls.validator_to_prompt(schema)

        # Return the XML as a string.
        return ET.tostring(schema.root, encoding="unicode", method="xml")
