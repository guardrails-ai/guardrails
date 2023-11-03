import json
import logging
import pprint
import re
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import pydantic
from lxml import etree as ET
from typing_extensions import Self

from guardrails import validator_service
from guardrails.datatypes import Choice, DataType, Object, String
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
from guardrails.utils.parsing_utils import get_template_variables
from guardrails.utils.reask_utils import (
    FieldReAsk,
    NonParseableReAsk,
    SkeletonReAsk,
    gather_reasks,
    get_pruned_tree,
    prune_obj_for_reasking,
)
from guardrails.utils.xml_utils import cast_xml_to_string
from guardrails.validator_base import (
    FailResult,
    Validator,
    check_refrain_in_dict,
    filter_in_dict,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FormatAttr(pydantic.BaseModel):
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
    format: Optional[str]

    # The on-fail handlers.
    on_fail_handlers: Dict[str, str]

    # The validator arguments.
    validator_args: Dict[str, List[Any]]

    # The validators.
    validators: List[Validator]

    # The unregistered validators.
    unregistered_validators: List[str]

    @property
    def empty(self) -> bool:
        """Return True if the format attribute is empty, False otherwise."""
        return self.format is None

    @classmethod
    def from_element(
        cls, element: ET._Element, tag: str, strict: bool = False
    ) -> "FormatAttr":
        """Create a FormatAttr object from an XML element.

        Args:
            element (ET._Element): The XML element.

        Returns:
            A FormatAttr object.
        """
        format_str = element.get("format")
        if format_str is None:
            return cls(
                format=None,
                on_fail_handlers={},
                validator_args={},
                validators=[],
                unregistered_validators=[],
            )

        validator_args = cls.parse(format_str)

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
            format=format_str,
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
            validator_name, args = FormatAttr.parse_token(token)
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

            # Create the validator.
            _validators.append(validator(*args, on_fail=on_fail))

        return _validators, _unregistered_validators

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

    reask_prompt_vars: Set[str]

    def __init__(
        self,
        schema: DataType,
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ) -> None:
        self.root_datatype = schema

        # Setup reask templates
        self.check_valid_reask_prompt(reask_prompt_template)
        if reask_prompt_template is not None:
            self._reask_prompt_template = Prompt(reask_prompt_template)
        else:
            self._reask_prompt_template = None
        if reask_instructions_template is not None:
            self._reask_instructions_template = Instructions(
                reask_instructions_template
            )
        else:
            self._reask_instructions_template = None

    @classmethod
    def from_element(
        cls,
        root: ET._Element,
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ) -> Self:
        """Create a schema from an XML element."""
        raise NotImplementedError

    def __repr__(self) -> str:
        # FIXME make sure this is pretty
        return f"{self.__class__.__name__}({pprint.pformat(self.root_datatype)})"

    @property
    def reask_prompt_template(self) -> Optional[Prompt]:
        return self._reask_prompt_template

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
        variables = get_template_variables(reask_prompt)
        assert set(variables) == self.reask_prompt_vars


class JsonSchema(Schema):
    reask_prompt_vars = {"previous_response", "output_schema"}

    def __init__(
        self,
        schema: Object,
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ) -> None:
        super().__init__(
            schema,
            reask_prompt_template=reask_prompt_template,
            reask_instructions_template=reask_instructions_template,
        )
        self.root_datatype = schema

    def get_reask_setup(
        self,
        reasks: List[FieldReAsk],
        original_response: Any,
        use_full_schema: bool,
        prompt_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple["Schema", Prompt, Instructions]:
        root = deepcopy(self.root_datatype)

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
            np_reask: NonParseableReAsk = original_response
            reask_value = np_reask.incorrect_value
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

                # Get the pruned tree so that it only contains ReAsk objects
                pruned_tree = get_pruned_tree(root, reasks)
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
            previous_response=json.dumps(
                reask_value, indent=2, default=reask_decoder, ensure_ascii=False
            ),
            output_schema=pruned_tree_string,
            **(prompt_params or {}),
        )

        instructions = self._reask_instructions_template
        if instructions is None:
            instructions = Instructions(constants["high_level_json_instructions"])
        instructions = instructions.format(**(prompt_params or {}))

        return pruned_tree_schema, prompt, instructions

    @classmethod
    def from_element(
        cls,
        root: ET._Element,
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ) -> Self:
        strict = False
        if "strict" in root.attrib and root.attrib["strict"] == "true":
            strict = True

        schema = Object.from_xml(root, strict=strict)

        return cls(
            schema,
            reask_prompt_template=reask_prompt_template,
            reask_instructions_template=reask_instructions_template,
        )

    def parse(
        self, output: str
    ) -> Tuple[Union[Optional[Dict], NonParseableReAsk], Optional[Exception]]:
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
            self.root_datatype,
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

        validation = self.root_datatype.collect_validation(
            key="",
            value=validated_response,
            schema=validated_response,
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
            self.root_datatype,
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

        # FIXME make the top-level validation key-invariant
        validation = self.root_datatype.collect_validation(
            key="",
            value=validated_response,
            schema=validated_response,
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
        schema: String,
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ) -> None:
        super().__init__(
            schema,
            reask_prompt_template=reask_prompt_template,
            reask_instructions_template=reask_instructions_template,
        )
        self.root_datatype = schema

    @classmethod
    def from_element(
        cls,
        root: ET._Element,
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ) -> Self:
        if len(root) != 0:
            raise ValueError("String output schemas must not have children.")

        strict = False
        if "strict" in root.attrib and root.attrib["strict"] == "true":
            strict = True

        schema = String.from_xml(root, strict=strict)

        return cls(
            schema=schema,
            reask_prompt_template=reask_prompt_template,
            reask_instructions_template=reask_instructions_template,
        )

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

        instructions = self._reask_instructions_template
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

        # FIXME instead of writing the validation infrastructure for dicts (JSON),
        #  make it more structure-invariant
        dummy_key = "string"
        validation = self.root_datatype.collect_validation(
            key=dummy_key,
            value=data,
            schema={
                dummy_key: data,
            },
        )

        validated_response, metadata = validator_service.validate(
            value=data,
            metadata=metadata,
            validator_setup=validation,
            validation_logs=validation_logs,
        )

        validated_response = {dummy_key: validated_response}

        if check_refrain_in_dict(validated_response):
            # If the data contains a `Refain` value, we return an empty
            # dictionary.
            logger.debug("Refrain detected.")
            validated_response = {}

        # Remove all keys that have `Filter` values.
        validated_response = filter_in_dict(validated_response)

        if dummy_key in validated_response:
            return validated_response[dummy_key]
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

        dummy_key = "string"
        validation = self.root_datatype.collect_validation(
            key=dummy_key,
            value=data,
            schema={
                dummy_key: data,
            },
        )

        validated_response, metadata = await validator_service.async_validate(
            value=data,
            metadata=metadata,
            validator_setup=validation,
            validation_logs=validation_logs,
        )

        validated_response = {dummy_key: validated_response}

        if check_refrain_in_dict(validated_response):
            # If the data contains a `Refain` value, we return an empty
            # dictionary.
            logger.debug("Refrain detected.")
            validated_response = {}

        # Remove all keys that have `Filter` values.
        validated_response = filter_in_dict(validated_response)

        if dummy_key in validated_response:
            return validated_response[dummy_key]
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
        obj = self.root_datatype
        schema = ""
        if obj.description is not None:
            schema += (
                "Here's a description of what I want you to generate: "
                f"{obj.description}"
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
    def datatypes_to_xml(
        dt: DataType,
        root: Optional[ET._Element] = None,
        override_tag_name: Optional[str] = None,
    ) -> ET._Element:
        """Recursively convert the datatypes to XML elements."""
        if root is None:
            tagname = override_tag_name or dt.tag
            el = ET.Element(tagname)
        else:
            el = ET.SubElement(root, dt.tag)

        if dt.name:
            el.attrib["name"] = dt.name

        if dt.description:
            el.attrib["description"] = dt.description

        if dt.format_attr:
            format_prompt = dt.format_attr.to_prompt()
            if format_prompt:
                el.attrib["format"] = format_prompt

        if dt.optional:
            el.attrib["required"] = "false"

        if isinstance(dt, Choice):
            el.attrib["discriminator"] = dt.discriminator_key

        for child in dt._children.values():
            Schema2Prompt.datatypes_to_xml(child, el)

        return el

    @classmethod
    def default(cls, schema: JsonSchema) -> str:
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
        schema_object = schema.root_datatype

        # Remove validators with arguments.
        root = cls.datatypes_to_xml(schema_object, override_tag_name="output")

        # Return the XML as a string that is
        ET.indent(root, space="    ")
        return ET.tostring(
            root,
            encoding="unicode",
            method="xml",
            pretty_print=True,
        )
