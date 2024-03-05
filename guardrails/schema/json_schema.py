import json
from copy import deepcopy
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
)

from lxml import etree as ET
from pydantic import BaseModel
from typing_extensions import Self

from guardrails import validator_service
from guardrails.classes.history import Iteration
from guardrails.datatypes import Choice, DataType
from guardrails.datatypes import List as ListDataType
from guardrails.datatypes import Object
from guardrails.llm_providers import (
    AsyncOpenAICallable,
    AsyncOpenAIChatCallable,
    OpenAICallable,
    OpenAIChatCallable,
    PromptCallableBase,
)
from guardrails.logger import logger
from guardrails.prompt import Instructions, Prompt
from guardrails.schema.schema import Schema
from guardrails.utils.constants import constants
from guardrails.utils.json_utils import (
    extract_json_from_ouput,
    verify_schema_against_json,
)
from guardrails.utils.pydantic_utils import convert_pydantic_model_to_datatype
from guardrails.utils.reask_utils import (
    FieldReAsk,
    NonParseableReAsk,
    ReAsk,
    SkeletonReAsk,
    gather_reasks,
    get_pruned_tree,
    prune_obj_for_reasking,
)
from guardrails.utils.safe_get import safe_get
from guardrails.utils.telemetry_utils import trace_validation_result
from guardrails.validator_base import FailResult, check_refrain, filter_in_schema
from guardrails.validatorsattr import ValidatorsAttr


class JsonSchema(Schema):
    reask_prompt_vars = {"previous_response", "output_schema", "json_example"}

    def __init__(
        self,
        schema: Union[Object, ListDataType],
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
        reasks: List[ReAsk],
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
            np_reask: NonParseableReAsk = next(
                r for r in reasks if isinstance(r, NonParseableReAsk)
            )
            # This is correct
            reask_value = np_reask.incorrect_value
        elif is_skeleton_reask:
            pruned_tree_schema = self

            reask_prompt_template = self.reask_prompt_template
            if reask_prompt_template is None:
                reask_prompt_template = Prompt(
                    constants["high_level_skeleton_reask_prompt"]
                    + constants["json_suffix_with_structure_example"]
                )

            # This is incorrect
            # This should be the parsed output
            reask_value = original_response
        else:
            if use_full_schema:
                # This is incorrect
                # This should be the parsed output
                reask_value = original_response
                # Don't prune the tree if we're reasking with pydantic model
                # (and openai function calling)
                pruned_tree_schema = self
            else:
                # This is correct
                reask_value = prune_obj_for_reasking(original_response)

                # Get the pruned tree so that it only contains ReAsk objects
                field_reasks = [r for r in reasks if isinstance(r, FieldReAsk)]
                pruned_tree = get_pruned_tree(root, field_reasks)
                pruned_tree_schema = type(self)(pruned_tree)

            reask_prompt_template = self.reask_prompt_template
            if reask_prompt_template is None:
                reask_prompt_template = Prompt(
                    constants["high_level_json_reask_prompt"]
                    + constants["json_suffix_without_examples"]
                )

        pruned_tree_string = pruned_tree_schema.transpile()
        json_example = json.dumps(
            pruned_tree_schema.root_datatype.get_example(),
            indent=2,
        )

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
            json_example=json_example,
            **(prompt_params or {}),
        )

        instructions = self.reask_instructions_template
        if instructions is None:
            instructions = Instructions(constants["high_level_json_instructions"])
        instructions = instructions.format(**(prompt_params or {}))

        return pruned_tree_schema, prompt, instructions

    @classmethod
    def from_xml(
        cls,
        root: ET._Element,
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ) -> Self:
        strict = False
        if "strict" in root.attrib and root.attrib["strict"] == "true":
            strict = True

        schema_type = root.attrib["type"] if "type" in root.attrib else "object"

        if schema_type == "list":
            schema = ListDataType.from_xml(root, strict=strict)
        else:
            schema = Object.from_xml(root, strict=strict)

        return cls(
            schema,
            reask_prompt_template=reask_prompt_template,
            reask_instructions_template=reask_instructions_template,
        )

    @classmethod
    def from_pydantic(
        cls,
        model: Union[Type[BaseModel], Type[List[Type[BaseModel]]]],
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ) -> Self:
        strict = False

        type_origin = get_origin(model)

        if type_origin == list:
            item_types = get_args(model)
            if len(item_types) > 1:
                raise ValueError("List data type must have exactly one child.")
            item_type = safe_get(item_types, 0)
            if not item_type or not issubclass(item_type, BaseModel):
                raise ValueError("List item type must be a Pydantic model.")
            item_schema = convert_pydantic_model_to_datatype(item_type, strict=strict)
            children = {"item": item_schema}
            validators_attr = ValidatorsAttr.from_validators(
                [], ListDataType.tag, strict
            )
            schema = ListDataType(children, validators_attr, False, None, None)
        else:
            pydantic_model = cast(Type[BaseModel], model)
            schema = convert_pydantic_model_to_datatype(pydantic_model, strict=strict)

        return cls(
            schema,
            reask_prompt_template=reask_prompt_template,
            reask_instructions_template=reask_instructions_template,
        )

    def parse(
        self, output: str, **kwargs
    ) -> Tuple[
        Union[Optional[Dict], NonParseableReAsk, str],
        Union[Optional[Exception], str, bool, None],
    ]:
        if kwargs.get("stream", False):
            # Do expected behavior for StreamRunner
            # 1. Check if the fragment is valid JSON
            verified = kwargs.get("verified", set())
            is_valid_fragment = self.is_valid_fragment(output, verified)
            if not is_valid_fragment:
                return output, True

            # 2. Parse the fragment
            parsed_fragment, parsing_error = self.parse_fragment(output)
            return parsed_fragment, parsing_error

        # Else do expected behavior for Runner
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

    def is_valid_fragment(self, fragment: str, verified: set) -> bool:
        """Check if the fragment is a somewhat valid JSON."""

        # Strip fragment of whitespaces and newlines
        # to avoid duplicate checks
        text = fragment.strip(" \n")

        # Check if text is already verified
        if text in verified:
            return False

        # Check if text is valid JSON
        try:
            json.loads(text)
            verified.add(text)
            return True
        except ValueError as e:
            error_msg = str(e)
            # Check if error is due to missing comma
            if "Expecting ',' delimiter" in error_msg:
                verified.add(text)
                return True
            return False

    def parse_fragment(self, fragment: str):
        """Parse the fragment into a dict."""

        # Complete the JSON fragment to handle missing brackets
        # Stack to keep track of opening brackets
        stack = []

        # Process each character in the string
        for char in fragment:
            if char in "{[":
                # Push opening brackets onto the stack
                stack.append(char)
            elif char in "}]":
                # Pop from stack if matching opening bracket is found
                if stack and (
                    (char == "}" and stack[-1] == "{")
                    or (char == "]" and stack[-1] == "[")
                ):
                    stack.pop()

        # Add the necessary closing brackets in reverse order
        while stack:
            opening_bracket = stack.pop()
            if opening_bracket == "{":
                fragment += "}"
            elif opening_bracket == "[":
                fragment += "]"

        # Parse the fragment
        try:
            parsed_fragment = json.loads(fragment)
            return parsed_fragment, None
        except ValueError as e:
            return fragment, str(e)

    def validate(
        self,
        iteration: Iteration,
        data: Optional[Dict[str, Any]],
        metadata: Dict,
        attempt_number: int = 0,
        **kwargs,
    ) -> Any:
        """Validate a dictionary of data against the schema.

        Args:
            data: The data to validate.

        Returns:
            The validated data.
        """
        if data is None:
            return None

        validated_response = deepcopy(data)

        if not verify_schema_against_json(
            self.root_datatype,
            validated_response,
            prune_extra_keys=True,
            coerce_types=True,
            validate_subschema=kwargs.get("validate_subschema", False),
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

        validated_response, metadata = validator_service.validate(
            value=validated_response,
            metadata=metadata,
            validator_setup=validation,
            iteration=iteration,
        )

        if check_refrain(validated_response):
            # If the data contains a `Refrain` value, we return an empty
            # dictionary.
            logger.debug("Refrain detected.")
            validated_response = {}

        # Remove all keys that have `Filter` values.
        validated_response = filter_in_schema(validated_response)

        # TODO: Capture error messages once Top Level error handling is merged in
        trace_validation_result(
            validation_logs=iteration.validator_logs, attempt_number=attempt_number
        )

        return validated_response

    async def async_validate(
        self,
        iteration: Iteration,
        data: Optional[Dict[str, Any]],
        metadata: Dict,
        attempt_number: int = 0,
    ) -> Any:
        """Validate a dictionary of data against the schema.

        Args:
            data: The data to validate.

        Returns:
            The validated data.
        """
        if data is None:
            return None

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

        validated_response, metadata = await validator_service.async_validate(
            value=validated_response,
            metadata=metadata,
            validator_setup=validation,
            iteration=iteration,
        )

        if check_refrain(validated_response):
            # If the data contains a `Refain` value, we return an empty
            # dictionary.
            logger.debug("Refrain detected.")
            validated_response = {}

        # Remove all keys that have `Filter` values.
        validated_response = filter_in_schema(validated_response)

        # TODO: Capture error messages once Top Level error handling is merged in
        trace_validation_result(
            validation_logs=iteration.validator_logs, attempt_number=attempt_number
        )

        return validated_response

    def introspect(self, data: Any) -> Tuple[List[ReAsk], Union[Dict, List, None]]:
        if isinstance(data, SkeletonReAsk):
            return [data], None
        elif isinstance(data, NonParseableReAsk):
            return [data], None
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

        if dt.validators_attr:
            format_prompt = dt.validators_attr.to_prompt()
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
