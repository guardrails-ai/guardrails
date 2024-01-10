from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from lxml import etree as ET
from typing_extensions import Self

from guardrails import validator_service
from guardrails.classes.history import Iteration
from guardrails.datatypes import String
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
from guardrails.utils.reask_utils import FieldReAsk, ReAsk
from guardrails.utils.telemetry_utils import trace_validation_result
from guardrails.validator_base import (
    ValidatorSpec,
    check_refrain_in_dict,
    filter_in_dict,
)


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
    def from_xml(
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

    @classmethod
    def from_string(
        cls,
        validators: Sequence[ValidatorSpec],
        description: Optional[str] = None,
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ):
        strict = False

        schema = String.from_string_rail(
            validators, description=description, strict=strict
        )

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

        instructions = self.reask_instructions_template
        if instructions is None:
            instructions = Instructions("You are a helpful assistant.")
        instructions = instructions.format(**(prompt_params or {}))

        return self, prompt, instructions

    def parse(self, output: str, **kwargs) -> Tuple[Any, Optional[Exception]]:
        return output, None

    def validate(
        self,
        iteration: Iteration,
        data: Any,
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

        if not isinstance(data, str):
            raise TypeError(f"Argument `data` must be a string, not {type(data)}.")

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
            iteration=iteration,
        )

        validated_response = {dummy_key: validated_response}

        if check_refrain_in_dict(validated_response):
            # If the data contains a `Refain` value, we return an empty
            # dictionary.
            logger.debug("Refrain detected.")
            validated_response = {}

        # Remove all keys that have `Filter` values.
        validated_response = filter_in_dict(validated_response)

        trace_validation_result(
            validation_logs=iteration.validator_logs, attempt_number=attempt_number
        )

        if dummy_key in validated_response:
            return validated_response[dummy_key]
        return None

    async def async_validate(
        self,
        iteration: Iteration,
        data: Any,
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

        if not isinstance(data, str):
            raise TypeError(f"Argument `data` must be a string, not {type(data)}.")

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
            iteration=iteration,
        )

        validated_response = {dummy_key: validated_response}

        if check_refrain_in_dict(validated_response):
            # If the data contains a `Refain` value, we return an empty
            # dictionary.
            logger.debug("Refrain detected.")
            validated_response = {}

        # Remove all keys that have `Filter` values.
        validated_response = filter_in_dict(validated_response)

        trace_validation_result(
            validation_logs=iteration.validator_logs, attempt_number=attempt_number
        )

        if dummy_key in validated_response:
            return validated_response[dummy_key]
        return None

    def introspect(
        self, data: Union[ReAsk, Optional[str]]
    ) -> Tuple[List[FieldReAsk], Optional[str]]:
        if isinstance(data, FieldReAsk):
            return [data], None
        return [], data  # type: ignore

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
        if not obj.validators_attr.empty:
            schema += (
                "\n\nYour generated response should satisfy the following properties:"
            )
            for validator in obj.validators_attr.validators:
                schema += f"\n- {validator.to_prompt()}"

        schema += "\n\nDon't talk; just go."
        return schema
