import re
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import lxml.etree as ET
import pydantic

from guardrails.utils.xml_utils import cast_xml_to_string
from guardrails.validator_base import Validator, ValidatorSpec


class ValidatorsAttr(pydantic.BaseModel):
    """Class for parsing and manipulating the `format` attribute of an element.

    The format attribute is a string that contains semi-colon separated
    validators e.g. "valid-url; is-reachable". Each validator is itself either:
    - the name of an parameter-less validator, e.g. "valid-url"
    - the name of a validator with parameters, separated by a colon with a
        space-separated list of parameters, e.g. "is-in: 1 2 3"

    Parameters can either be written in plain text, or in python expressions
    enclosed in curly braces. For example, the following are all valid:
    - "is-in: 1 2 3"
    - "is-in: {1} {2} {3}"
    - "is-in: {1 + 2} {2 + 3} {3 + 4}"
    """

    class Config:
        arbitrary_types_allowed = True

    # The format attribute string.
    validators_spec: Optional[str]

    # The on-fail handlers.
    on_fail_handlers: Mapping[str, Union[str, Callable]]

    # The validator arguments.
    validator_args: Mapping[str, Union[Dict[str, Any], List[Any]]]

    # The validators.
    validators: List[Validator]

    # The unregistered validators.
    unregistered_validators: List[str]

    @property
    def empty(self) -> bool:
        """Return True if the format attribute is empty, False otherwise."""
        return not self.validators and not self.unregistered_validators

    @classmethod
    def from_validators(
        cls,
        validators: Sequence[ValidatorSpec],
        tag: str,
        strict: bool = False,
    ):
        validators_with_args = {}
        on_fails = {}
        for val in validators:
            # must be either a tuple with two elements or a gd.Validator
            if isinstance(val, Validator):
                # `validator` is of type gd.Validator, use the to_xml_attrib method
                validator_name = val.rail_alias
                validator_args = val.get_args()
                validators_with_args[validator_name] = validator_args
                # Set the on-fail attribute based on the on_fail value
                if val.on_fail_descriptor == "custom":
                    on_fail = val.on_fail_method
                else:
                    on_fail = val.on_fail_descriptor
                on_fails[val.rail_alias] = on_fail
            elif isinstance(val, tuple) and len(val) == 2:
                validator, on_fail = val
                if isinstance(validator, Validator):
                    # `validator` is of type gd.Validator, use the to_xml_attrib method
                    validator_name = validator.rail_alias
                    validator_args = validator.get_args()
                    validators_with_args[validator_name] = validator_args
                    # Set the on-fail attribute based on the on_fail value
                    on_fails[validator.rail_alias] = on_fail
                elif isinstance(validator, str):
                    # `validator` is a string, use it as the validator prompt
                    if ":" in validator:
                        parts = validator.split(":", 1)
                        validator_name = parts[0].strip()
                        validator_args = [
                            arg.strip() for arg in parts[1].split() if len(parts) > 1
                        ]
                    else:
                        validator_name = validator
                        validator_args = []
                    validators_with_args[validator_name] = validator_args
                    on_fails[validator_name] = on_fail
                elif isinstance(validator, Callable):
                    # `validator` is a callable, use it as the validator prompt
                    if not hasattr(validator, "rail_alias"):
                        raise ValueError(
                            f"Validator {validator.__name__} must be registered with "
                            f"the gd.register_validator decorator"
                        )
                    validator_name = validator.rail_alias
                    validator_args = []
                    validators_with_args[validator_name] = validator_args
                    on_fails[validator.rail_alias] = on_fail
                else:
                    raise ValueError(
                        f"Validator tuple {val} must be a (validator, on_fail) tuple, "
                        f"where the validator is a string or a callable"
                    )
            else:
                raise ValueError(
                    f"Validator {val} must be a (validator, on_fail) tuple or "
                    f"Validator class instance"
                )

        registered_validators, unregistered_validators = cls.get_validators(
            validator_args=validators_with_args,
            tag=tag,
            on_fail_handlers=on_fails,
            strict=strict,
        )

        return cls(
            validators_spec=None,
            on_fail_handlers=on_fails,
            validator_args=validators_with_args,
            validators=registered_validators,
            unregistered_validators=unregistered_validators,
        )

    @classmethod
    def from_xml(
        cls, element: ET._Element, tag: str, strict: bool = False
    ) -> "ValidatorsAttr":
        """Create a ValidatorsAttr object from an XML element.

        Args:
            element (ET._Element): The XML element.

        Returns:
            A ValidatorsAttr object.
        """
        validators_str = element.get("validators")
        format_str = element.get("format")
        if format_str is not None:
            warnings.warn(
                "Attribute `format` is deprecated and will be removed in 0.4.x. "
                "Use `validators` instead.",
                DeprecationWarning,
            )
            validators_str = format_str

        if validators_str is None:
            return cls(
                validators_spec=None,
                on_fail_handlers={},
                validator_args={},
                validators=[],
                unregistered_validators=[],
            )

        validator_args = cls.parse(validators_str)

        on_fail_handlers = {}
        for key, value in element.attrib.items():
            key = cast_xml_to_string(key)
            if key.startswith("on-fail-"):
                on_fail_handler_name = key[len("on-fail-") :]
                on_fail_handler = value
                on_fail_handlers[on_fail_handler_name] = on_fail_handler

        validators, unregistered_validators = cls.get_validators(
            validator_args=validator_args,
            tag=tag,
            on_fail_handlers=on_fail_handlers,
            strict=strict,
        )

        return cls(
            validators_spec=validators_str,
            on_fail_handlers=on_fail_handlers,
            validator_args=validator_args,
            validators=validators,
            unregistered_validators=unregistered_validators,
        )

    @staticmethod
    def parse_token(token: str) -> Tuple[str, List[Any]]:
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

    @staticmethod
    def parse(format_string: str) -> Dict[str, List[Any]]:
        """Parse the format attribute into a dictionary of validators.

        Returns:
            A dictionary of validators, where the key is the validator name, and
            the value is a list of arguments.
        """
        # Split the format attribute into tokens: each is a validator.
        # Then, parse each token into a validator name and a list of parameters.
        pattern = re.compile(r";(?![^{}]*})")
        tokens = re.split(pattern, format_string)
        tokens = list(filter(None, tokens))

        validators = {}
        for token in tokens:
            # Parse the token into a validator name and a list of parameters.
            validator_name, args = ValidatorsAttr.parse_token(token)
            validators[validator_name] = args

        return validators

    @staticmethod
    def get_validators(
        validator_args: Dict[str, List[Any]],
        tag: str,
        on_fail_handlers: Dict[str, str],
        strict: bool = False,
    ) -> Tuple[List[Validator], List[str]]:
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
        from guardrails.validator_base import types_to_validators, validators_registry

        _validators = []
        _unregistered_validators = []
        for validator_name, args in validator_args.items():
            # Check if the validator is registered for this element.
            # The validators in `format` that are not registered for this element
            # will be ignored (with an error or warning, depending on the value of
            # `strict`), and the registered validators will be returned.
            if validator_name not in types_to_validators[tag]:
                if strict:
                    raise ValueError(
                        f"Validator {validator_name} is not valid for"
                        f" element {tag}."
                    )
                else:
                    warnings.warn(
                        f"Validator {validator_name} is not valid for"
                        f" element {tag}."
                    )
                    _unregistered_validators.append(validator_name)
                continue

            validator = validators_registry[validator_name]

            # See if the formatter has an associated on_fail method.
            on_fail = on_fail_handlers.get(validator_name, None)
            # TODO(shreya): Load the on_fail method.
            # This method should be loaded from an optional script given at the
            # beginning of a rail file.

            # Use inline import to avoid circular dependency
            from guardrails.validators import ValidChoices

            # Create the validator.
            if isinstance(args, list):
                # TODO: Handle different args type properly
                if validator == ValidChoices:
                    if isinstance(args[0], list):
                        v = validator(args[0], on_fail=on_fail)
                    else:
                        v = validator(args, on_fail=on_fail)
                else:
                    v = validator(*args, on_fail=on_fail)
            elif isinstance(args, dict):
                v = validator(**args, on_fail=on_fail)
            else:
                raise ValueError(
                    f"Validator {validator_name} has invalid arguments: {args}."
                )
            _validators.append(v)

        return _validators, _unregistered_validators

    def to_prompt(self, with_keywords: bool = True) -> str:
        """Convert the format string to another string representation for use
        in prompting. Uses the validators' to_prompt method in order to
        construct the string to use in prompting.

        For example, the format string "valid-url; other-validator: 1.0
        {1 + 2}" will be converted to "valid-url other-validator:
        arg1=1.0 arg2=3".
        """
        if self.empty:
            return ""
        # Use the validators' to_prompt method to convert the format string to
        # another string representation.
        prompt = "; ".join([v.to_prompt(with_keywords) for v in self.validators])
        unreg_prompt = "; ".join(self.unregistered_validators)
        if prompt and unreg_prompt:
            prompt += f"; {unreg_prompt}"
        elif unreg_prompt:
            prompt += unreg_prompt
        return prompt
