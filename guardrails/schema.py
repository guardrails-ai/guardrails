import json
import logging
import pprint
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass
from string import Formatter
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from lxml import etree as ET

from guardrails import validator_service
from guardrails.datatypes import DataType, String
from guardrails.llm_providers import (
    AsyncOpenAICallable,
    AsyncOpenAIChatCallable,
    OpenAICallable,
    OpenAIChatCallable,
    PromptCallableBase,
)
from guardrails.prompt import Instructions, Prompt
from guardrails.utils.constants import constants
from guardrails.utils.json_utils import (
    extract_json_from_ouput,
    verify_schema_against_json,
)
from guardrails.utils.logs_utils import FieldValidationLogs, GuardLogs
from guardrails.utils.reask_utils import (
    FieldReAsk,
    NonParseableReAsk,
    SkeletonReAsk,
    gather_reasks,
    get_pruned_tree,
    get_reasks_by_element,
    prune_obj_for_reasking,
)
from guardrails.validator_service import FieldValidation
from guardrails.validators import (
    FailResult,
    Validator,
    check_refrain_in_dict,
    filter_in_dict,
)

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
        space-separated list of parameters, e.g. "is-in: 1 2 3"

    Parameters can either be written in plain text, or in python expressions
    enclosed in curly braces. For example, the following are all valid:
    - "is-in: 1 2 3"
    - "is-in: {1} {2} {3}"
    - "is-in: {1 + 2} {2 + 3} {3 + 4}"
    """

    # The format attribute string.
    format: Optional[str] = None

    # The XML element that this format attribute is associated with.
    element: Optional[ET._Element] = None

    @property
    def empty(self) -> bool:
        """Return True if the format attribute is empty, False otherwise."""
        return self.format is None

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

    @classmethod
    def parse_token(cls, token: str) -> Tuple[str, List[Any]]:
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

    @property
    def unregistered_validators(self) -> List[str]:
        """Get the list of validators from the format attribute that are not
        registered for this element."""
        try:
            return getattr(self, "_unregistered_validators")
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
        _unregistered_validators = []
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
                    _unregistered_validators.append(validator_name)
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
        self._unregistered_validators = _unregistered_validators
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
        prompt = "; ".join([v.to_prompt(with_keywords) for v in self.validators])
        unreg_prompt = "; ".join(self.unregistered_validators)
        if prompt and unreg_prompt:
            prompt += f"; {unreg_prompt}"
        elif unreg_prompt:
            prompt += unreg_prompt
        return prompt


class Schema:
    """Schema class that holds a _schema attribute."""

    def __init__(
        self,
        root: Optional[ET._Element] = None,
        schema: Optional[Dict[str, DataType]] = None,
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ) -> None:
        # Setup schema
        if schema is None:
            schema = {}
        self._schema = SimpleNamespace(**schema)

        # Setup root
        self.root = root
        if root is not None:
            self.setup_schema(root)

        # Setup reask templates
        self.check_valid_reask_prompt(reask_prompt_template)
        self._reask_prompt_template = reask_prompt_template
        if reask_prompt_template is not None:
            reask_prompt_template = Prompt(reask_prompt_template)
        self.reask_instructions_template = reask_instructions_template
        if reask_instructions_template is not None:
            reask_instructions_template = Prompt(reask_instructions_template)

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert the schema to a dictionary."""
        return vars(self._schema)

    @property
    def parsed_rail(self) -> Optional[ET._Element]:
        return self.root

    @property
    def reask_prompt_template(self) -> Optional[Prompt]:
        return self._reask_prompt_template

    def setup_schema(self, root: ET._Element) -> None:
        """Parse the schema specification.

        Args:
            root: The root element of the schema specification.
        """
        raise NotImplementedError

    def validate(self, guard_logs: GuardLogs, data: Any, metadata: Dict) -> Any:
        """Validate a dictionary of data against the schema.

        Args:
            data: The data to validate.

        Returns:
            The validated data.
        """
        raise NotImplementedError

    async def async_validate(
        self, guard_logs: GuardLogs, data: Any, metadata: Dict
    ) -> Any:
        """Asynchronously validate a dictionary of data against the schema.

        Args:
            data: The data to validate.

        Returns:
            The validated data.
        """
        raise NotImplementedError

    def transpile(self, method: str = "default") -> str:
        """Convert the XML schema to a string that is used for prompting a
        large language model.

        Returns:
            The prompt.
        """
        raise NotImplementedError

    def parse(self, output: str) -> Tuple[Any, Optional[Exception]]:
        """Parse the output from the large language model.

        Args:
            output: The output from the large language model.

        Returns:
            The parsed output, and the exception that was raised (if any).
        """
        raise NotImplementedError

    def introspect(self, data: Any) -> List[FieldReAsk]:
        """Inspect the data for reasks.

        Args:
            data: The data to introspect.

        Returns:
            A list of ReAsk objects.
        """
        raise NotImplementedError

    def get_reask_setup(
        self,
        reasks: List[FieldReAsk],
        original_response: Any,
        use_full_schema: bool,
        prompt_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple["Schema", Prompt, Instructions]:
        """Construct a schema for reasking, and a prompt for reasking.

        Args:
            reasks: List of tuples, where each tuple contains the path to the
                reasked element, and the ReAsk object (which contains the error
                message describing why the reask is necessary).
            original_response: The value that was returned from the API, with reasks.
            use_full_schema: Whether to use the full schema, or only the schema
                for the reasked elements.

        Returns:
            The schema for reasking, and the prompt for reasking.
        """
        raise NotImplementedError

    def preprocess_prompt(
        self,
        prompt_callable: PromptCallableBase,
        instructions: Optional[Instructions],
        prompt: Prompt,
    ):
        """Preprocess the instructions and prompt before sending it to the
        model.

        Args:
            prompt_callable: The callable to be used to prompt the model.
            instructions: The instructions to preprocess.
            prompt: The prompt to preprocess.
        """
        raise NotImplementedError

    def check_valid_reask_prompt(self, reask_prompt: Optional[str]) -> None:
        if reask_prompt is None:
            return

        # Check that the reask prompt has the correct variables
        variables = [t[1] for t in Formatter().parse(reask_prompt) if t[1] is not None]
        assert set(variables) == self.reask_prompt_vars


class JsonSchema(Schema):
    reask_prompt_vars = {"previous_response", "output_schema"}

    def get_reask_setup(
        self,
        reasks: List[FieldReAsk],
        original_response: Any,
        use_full_schema: bool,
        prompt_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple["Schema", Prompt, Instructions]:
        parsed_rail = deepcopy(self.root)

        is_skeleton_reask = not any(isinstance(reask, FieldReAsk) for reask in reasks)
        is_nonparseable_reask = any(
            isinstance(reask, NonParseableReAsk) for reask in reasks
        )

        if is_nonparseable_reask:
            pruned_tree_schema = self

            reask_prompt_template = self.reask_prompt_template
            if reask_prompt_template is None:
                reask_prompt_template = Prompt(
                    constants["high_level_json_parsing_reask_prompt"]
                    + constants["json_suffix_without_examples"]
                )
            reask_value = original_response
        elif is_skeleton_reask:
            pruned_tree_schema = self

            reask_prompt_template = self.reask_prompt_template
            if reask_prompt_template is None:
                reask_prompt_template = Prompt(
                    constants["high_level_skeleton_reask_prompt"]
                    + constants["json_suffix_without_examples"]
                )

            reask_value = original_response
        else:
            if use_full_schema:
                reask_value = original_response
                # Don't prune the tree if we're reasking with pydantic model
                # (and openai function calling)
                pruned_tree_schema = self
            else:
                reask_value = prune_obj_for_reasking(original_response)
                # Get the elements that are to be reasked
                reask_elements = get_reasks_by_element(reasks, parsed_rail)

                # Get the pruned tree so that it only contains ReAsk objects
                pruned_tree = get_pruned_tree(parsed_rail, list(reask_elements.keys()))
                pruned_tree_schema = type(self)(pruned_tree)

            reask_prompt_template = self.reask_prompt_template
            if reask_prompt_template is None:
                reask_prompt_template = Prompt(
                    constants["high_level_json_reask_prompt"]
                    + constants["json_suffix_without_examples"]
                )

        pruned_tree_string = pruned_tree_schema.transpile()

        def reask_decoder(obj):
            decoded = {}
            for k, v in obj.__dict__.items():
                if k in ["path"]:
                    continue
                if k == "fail_results":
                    k = "error_messages"
                    v = [result.error_message for result in v]
                decoded[k] = v
            return decoded

        prompt = reask_prompt_template.format(
            previous_response=json.dumps(reask_value, indent=2, default=reask_decoder),
            output_schema=pruned_tree_string,
            **(prompt_params or {}),
        )

        instructions = self.reask_instructions_template
        if instructions is None:
            instructions = Instructions(constants["high_level_json_instructions"])
        instructions = instructions.format(**(prompt_params or {}))

        return pruned_tree_schema, prompt, instructions

    def setup_schema(self, root: ET._Element) -> None:
        from guardrails.datatypes import registry as types_registry

        strict = False
        if "strict" in root.attrib and root.attrib["strict"] == "true":
            strict = True

        for child in root:
            if isinstance(child, ET._Comment):
                continue
            child_name = child.attrib["name"]
            child_data = types_registry[child.tag].from_xml(child, strict=strict)
            self[child_name] = child_data

    def parse(self, output: str) -> Tuple[Dict, Optional[Exception]]:
        # Try to get json code block from output.
        # Return error and reask if it is not parseable.
        parsed_output, error = extract_json_from_ouput(output)

        if error:
            reask = NonParseableReAsk(
                incorrect_value=output,
                fail_results=[
                    FailResult(
                        fix_value=None,
                        error_message="Output is not parseable as JSON",
                    )
                ],
            )
            return reask, error
        return parsed_output, None

    def validate(
        self,
        guard_logs: GuardLogs,
        data: Optional[Dict[str, Any]],
        metadata: Dict,
    ) -> Any:
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

        if not verify_schema_against_json(
            self.root,
            validated_response,
            prune_extra_keys=True,
            coerce_types=True,
        ):
            return SkeletonReAsk(
                incorrect_value=validated_response,
                fail_results=[
                    FailResult(
                        fix_value=None,
                        error_message="JSON does not match schema",
                    )
                ],
            )

        validation = FieldValidation(
            key="",
            value=validated_response,
            validators=[],
            children=[],
        )

        for field, value in validated_response.items():
            if field not in self:
                # This is an extra field that is not in the schema.
                # We remove it from the validated response.
                logger.debug(f"Field {field} not in schema.")
                continue

            logger.debug(f"Validating field {field} with value {value}.")

            field_validation = self[field].collect_validation(
                key=field,
                value=value,
                schema=validated_response,
            )
            validation.children.append(field_validation)

            logger.debug(
                f"Validated field {field} with value {validated_response[field]}."
            )

        validation_logs = FieldValidationLogs()
        guard_logs.field_validation_logs = validation_logs

        validated_response, metadata = validator_service.validate(
            value=validated_response,
            metadata=metadata,
            validator_setup=validation,
            validation_logs=validation_logs,
        )

        if check_refrain_in_dict(validated_response):
            # If the data contains a `Refain` value, we return an empty
            # dictionary.
            logger.debug("Refrain detected.")
            validated_response = {}

        # Remove all keys that have `Filter` values.
        validated_response = filter_in_dict(validated_response)

        return validated_response

    async def async_validate(
        self,
        guard_logs: GuardLogs,
        data: Optional[Dict[str, Any]],
        metadata: Dict,
    ) -> Any:
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

        if not verify_schema_against_json(
            self.root,
            validated_response,
            prune_extra_keys=True,
            coerce_types=True,
        ):
            return SkeletonReAsk(
                incorrect_value=validated_response,
                fail_results=[
                    FailResult(
                        fix_value=None,
                        error_message="JSON does not match schema",
                    )
                ],
            )

        validation = FieldValidation(
            key="",
            value=validated_response,
            validators=[],
            children=[],
        )

        for field, value in validated_response.items():
            if field not in self:
                # This is an extra field that is not in the schema.
                # We remove it from the validated response.
                logger.debug(f"Field {field} not in schema.")
                continue

            logger.debug(f"Validating field {field} with value {value}.")

            field_validation = self[field].collect_validation(
                key=field,
                value=value,
                schema=validated_response,
            )
            validation.children.append(field_validation)

            logger.debug(
                f"Validated field {field} with value {validated_response[field]}."
            )

        validation_logs = FieldValidationLogs()
        guard_logs.field_validation_logs = validation_logs

        validated_response, metadata = await validator_service.async_validate(
            value=validated_response,
            metadata=metadata,
            validator_setup=validation,
            validation_logs=validation_logs,
        )

        if check_refrain_in_dict(validated_response):
            # If the data contains a `Refain` value, we return an empty
            # dictionary.
            logger.debug("Refrain detected.")
            validated_response = {}

        # Remove all keys that have `Filter` values.
        validated_response = filter_in_dict(validated_response)

        return validated_response

    def introspect(self, data: Any) -> list:
        if isinstance(data, SkeletonReAsk):
            return [data]
        elif isinstance(data, NonParseableReAsk):
            return [data]
        return gather_reasks(data)

    def preprocess_prompt(
        self,
        prompt_callable: PromptCallableBase,
        instructions: Optional[Instructions],
        prompt: Prompt,
    ):
        if isinstance(prompt_callable, OpenAICallable) or isinstance(
            prompt_callable, AsyncOpenAICallable
        ):
            prompt.source += "\n\nJson Output:\n\n"
        if (
            isinstance(prompt_callable, OpenAIChatCallable)
            or isinstance(prompt_callable, AsyncOpenAIChatCallable)
        ) and not instructions:
            instructions = Instructions(
                "You are a helpful assistant, "
                "able to express yourself purely through JSON, "
                "strictly and precisely adhering to the provided XML schemas."
            )

        return instructions, prompt

    def transpile(self, method: str = "default") -> str:
        transpiler = getattr(Schema2Prompt, method)
        return transpiler(self)


class StringSchema(Schema):
    reask_prompt_vars = {"previous_response", "output_schema", "error_messages"}

    def __init__(
        self,
        root: ET._Element,
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ) -> None:
        self.string_key = "string"
        super().__init__(root)

        # Setup reask templates
        self._reask_prompt_template = reask_prompt_template
        if self._reask_prompt_template is not None:
            self._reask_prompt_template = Prompt(reask_prompt_template)
        self.reask_instructions_template = reask_instructions_template
        if self.reask_instructions_template is not None:
            self.reask_instructions_template = Prompt(self.reask_instructions_template)

    def setup_schema(self, root: ET._Element) -> None:
        if len(root) != 0:
            raise ValueError("String output schemas must not have children.")

        if "name" in root.attrib:
            self.string_key = root.attrib["name"]
        else:
            self.string_key = root.attrib["name"] = "string"

        # make root tag into a string tag
        root_string = ET.Element("string", root.attrib)
        self[self.string_key] = String.from_xml(root_string)

    def get_reask_setup(
        self,
        reasks: List[FieldReAsk],
        original_response: FieldReAsk,
        use_full_schema: bool,
        prompt_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Schema, Prompt, Instructions]:
        pruned_tree_string = self.transpile()

        reask_prompt_template = self.reask_prompt_template
        if reask_prompt_template is None:
            reask_prompt_template = Prompt(
                constants["high_level_string_reask_prompt"]
                + constants["complete_string_suffix"]
            )

        error_messages = "\n".join(
            [
                f"- {fail_result.error_message}"
                for reask in reasks
                for fail_result in reask.fail_results
            ]
        )

        prompt = reask_prompt_template.format(
            previous_response=original_response.incorrect_value,
            error_messages=error_messages,
            output_schema=pruned_tree_string,
            **(prompt_params or {}),
        )

        instructions = self.reask_instructions_template
        if instructions is None:
            instructions = Instructions("You are a helpful assistant.")
        instructions = instructions.format(**(prompt_params or {}))

        return self, prompt, instructions

    def parse(self, output: str) -> Tuple[Any, Optional[Exception]]:
        return output, None

    def validate(
        self,
        guard_logs: GuardLogs,
        data: Any,
        metadata: Dict,
    ) -> Any:
        """Validate a dictionary of data against the schema.

        Args:
            data: The data to validate.

        Returns:
            The validated data.
        """
        if data is None:
            return None

        if not isinstance(data, str):
            raise TypeError(f"Argument `data` must be a string, not {type(data)}.")

        validation_logs = FieldValidationLogs()
        guard_logs.field_validation_logs = validation_logs

        validation = self[self.string_key].collect_validation(
            key=self.string_key,
            value=data,
            schema={
                self.string_key: data,
            },
        )

        validated_response, metadata = validator_service.validate(
            value=data,
            metadata=metadata,
            validator_setup=validation,
            validation_logs=validation_logs,
        )

        validated_response = {self.string_key: validated_response}

        if check_refrain_in_dict(validated_response):
            # If the data contains a `Refain` value, we return an empty
            # dictionary.
            logger.debug("Refrain detected.")
            validated_response = {}

        # Remove all keys that have `Filter` values.
        validated_response = filter_in_dict(validated_response)

        if self.string_key in validated_response:
            return validated_response[self.string_key]
        return None

    async def async_validate(
        self,
        guard_logs: GuardLogs,
        data: Any,
        metadata: Dict,
    ) -> Any:
        """Validate a dictionary of data against the schema.

        Args:
            data: The data to validate.

        Returns:
            The validated data.
        """
        if data is None:
            return None

        if not isinstance(data, str):
            raise TypeError(f"Argument `data` must be a string, not {type(data)}.")

        validation_logs = FieldValidationLogs()
        guard_logs.field_validation_logs = validation_logs

        validation = self[self.string_key].collect_validation(
            key=self.string_key,
            value=data,
            schema={
                self.string_key: data,
            },
        )

        validated_response, metadata = await validator_service.async_validate(
            value=data,
            metadata=metadata,
            validator_setup=validation,
            validation_logs=validation_logs,
        )

        validated_response = {self.string_key: validated_response}

        if check_refrain_in_dict(validated_response):
            # If the data contains a `Refain` value, we return an empty
            # dictionary.
            logger.debug("Refrain detected.")
            validated_response = {}

        # Remove all keys that have `Filter` values.
        validated_response = filter_in_dict(validated_response)

        if self.string_key in validated_response:
            return validated_response[self.string_key]
        return None

    def introspect(self, data: Any) -> List[FieldReAsk]:
        if isinstance(data, FieldReAsk):
            return [data]
        return []

    def preprocess_prompt(
        self,
        prompt_callable: PromptCallableBase,
        instructions: Optional[Instructions],
        prompt: Prompt,
    ):
        if isinstance(prompt_callable, OpenAICallable) or isinstance(
            prompt_callable, AsyncOpenAICallable
        ):
            prompt.source += "\n\nString Output:\n\n"
        if (
            isinstance(prompt_callable, OpenAIChatCallable)
            or isinstance(prompt_callable, AsyncOpenAIChatCallable)
        ) and not instructions:
            instructions = Instructions(
                "You are a helpful assistant, expressing yourself through a string."
            )

        return instructions, prompt

    def transpile(self, method: str = "default") -> str:
        obj = self[self.string_key]
        schema = ""
        if "description" in obj.element.attrib:
            schema += (
                "Here's a description of what I want you to generate: "
                f"{obj.element.attrib['description']}"
            )
        if not obj.format_attr.empty:
            schema += (
                "\n\nYour generated response should satisfy the following properties:"
            )
            for validator in obj.format_attr.validators:
                schema += f"\n- {validator.to_prompt()}"

        schema += "\n\nDon't talk; just go."
        return schema


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
    def validator_to_prompt(root: ET.Element, schema_dict: Dict[str, DataType]) -> None:
        """Recursively remove all validator arguments in the `format`
        attribute."""

        def _inner(dt: DataType, el: ET._Element):
            if "format" in el.attrib:
                format = dt.format_attr.to_prompt()
                if len(format):
                    el.attrib["format"] = format
                else:
                    del el.attrib["format"]

            for _, dt_child, el_child in dt.iter(el):
                _inner(dt_child, el_child)

        for el_child in root:
            dt_child = schema_dict[el_child.attrib["name"]]
            _inner(dt_child, el_child)

    @staticmethod
    def pydantic_to_object(root: ET.Element, schema_dict: Dict[str, DataType]) -> None:
        """Recursively replace all pydantic elements with object elements."""
        from guardrails.datatypes import Pydantic

        def _inner(dt: DataType, el: ET._Element):
            if isinstance(dt, Pydantic):
                new_el = dt.to_object_element()
                el.getparent().replace(el, new_el)

            for _, dt_child, el_child in dt.iter(el):
                _inner(dt_child, el_child)

        for el_child in root:
            dt_child = schema_dict[el_child.attrib["name"]]
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
        root = deepcopy(schema.root)
        schema_dict = schema.to_dict()

        # Remove comments.
        cls.remove_comments(root)
        # Remove action attributes.
        cls.remove_on_fail_attributes(root)
        # Remove validators with arguments.
        cls.validator_to_prompt(root, schema_dict)
        # Replace pydantic elements with object elements.
        cls.pydantic_to_object(root, schema_dict)

        # Return the XML as a string that is
        ET.indent(root, space="    ")
        return ET.tostring(
            root,
            encoding="unicode",
            method="xml",
            pretty_print=True,
        )
