import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from eliot import add_destinations, start_action
from pydantic import BaseModel

from guardrails.datatypes import verify_metadata_requirements
from guardrails.llm_providers import AsyncPromptCallableBase, PromptCallableBase
from guardrails.prompt import Instructions, Prompt
from guardrails.schema import Schema
from guardrails.utils.logs_utils import GuardHistory, GuardLogs, GuardState, LLMResponse
from guardrails.utils.reask_utils import (
    FieldReAsk,
    NonParseableReAsk,
    ReAsk,
    reasks_to_dict,
    sub_reasks_with_fixed_values,
)

logger = logging.getLogger(__name__)
actions_logger = logging.getLogger(f"{__name__}.actions")
add_destinations(actions_logger.debug)


@dataclass
class Runner:
    """Runner class that calls an LLM API with a prompt, and performs input and
    output validation.

    This class will repeatedly call the API until the
    reask budget is exhausted, or the output is valid.

    Args:
        prompt: The prompt to use.
        api: The LLM API to call, which should return a string.
        input_schema: The input schema to use for validation.
        output_schema: The output schema to use for validation.
        num_reasks: The maximum number of times to reask the LLM in case of
            validation failure, defaults to 0.
        output: The output to use instead of calling the API, used in cases
            where the output is already known.
        guard_history: The guard history to use, defaults to an empty history.
    """

    instructions: Optional[Instructions]
    prompt: Prompt
    msg_history: Optional[List[Dict]]
    api: PromptCallableBase
    input_schema: Schema
    output_schema: Schema
    guard_state: GuardState
    num_reasks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    output: str = None
    reask_prompt: Optional[Prompt] = None
    reask_instructions: Optional[Instructions] = None
    guard_history: GuardHistory = field(
        default_factory=lambda: GuardHistory(history=[])
    )
    base_model: Optional[BaseModel] = None
    full_schema_reask: bool = False

    def _reset_guard_history(self):
        """Reset the guard history."""
        self.guard_history = GuardHistory(history=[])
        self.guard_state.push(self.guard_history)

    def __post_init__(self):
        if self.prompt:
            assert self.api, "Must provide an API if a prompt is provided."
            assert not self.output, "Cannot provide both a prompt and output."

        if isinstance(self.prompt, str):
            self.prompt = Prompt(
                self.prompt, output_schema=self.output_schema.transpile()
            )

        if isinstance(self.instructions, str):
            self.instructions = Instructions(
                self.instructions, output_schema=self.output_schema.transpile()
            )

        if self.msg_history is not None and len(self.msg_history):
            self.msg_history = copy.deepcopy(self.msg_history)
            msg_history = []
            for msg in self.msg_history:
                msg["content"] = Prompt(
                    msg["content"], output_schema=self.output_schema.transpile()
                )
                msg_history.append(msg)
            self.msg_history = msg_history

    def __call__(self, prompt_params: Dict = None) -> GuardHistory:
        """Execute the runner by repeatedly calling step until the reask budget
        is exhausted.

        Args:
            prompt_params: Parameters to pass to the prompt in order to
                generate the prompt string.

        Returns:
            The guard history.
        """
        # check if validator requirements are fulfilled
        missing_keys = verify_metadata_requirements(
            self.metadata, self.output_schema.to_dict().values()
        )
        if missing_keys:
            raise ValueError(
                f"Missing required metadata keys: {', '.join(missing_keys)}"
            )

        self._reset_guard_history()

        # Figure out if we need to include instructions in the prompt.
        include_instructions = not (
            self.instructions is None and self.msg_history is None
        )

        with start_action(
            action_type="run",
            instructions=self.instructions,
            prompt=self.prompt,
            api=self.api,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            num_reasks=self.num_reasks,
            metadata=self.metadata,
        ):
            instructions, prompt, msg_history, input_schema, output_schema = (
                self.instructions,
                self.prompt,
                self.msg_history,
                self.input_schema,
                self.output_schema,
            )
            for index in range(self.num_reasks + 1):
                # Run a single step.
                validated_output, reasks = self.step(
                    index=index,
                    api=self.api,
                    instructions=instructions,
                    prompt=prompt,
                    msg_history=msg_history,
                    prompt_params=prompt_params,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    output=self.output if index == 0 else None,
                )

                # Loop again?
                if not self.do_loop(index, reasks):
                    break
                # Get new prompt and output schema.
                prompt, instructions, output_schema, msg_history = self.prepare_to_loop(
                    reasks,
                    validated_output,
                    output_schema,
                    include_instructions=include_instructions,
                    prompt_params=prompt_params,
                )

            return self.guard_history

    def step(
        self,
        index: int,
        api: PromptCallableBase,
        instructions: Optional[Instructions],
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict]],
        prompt_params: Dict,
        input_schema: Schema,
        output_schema: Schema,
        output: str = None,
    ):
        guard_logs = GuardLogs()
        self.guard_history.push(guard_logs)
        """Run a full step."""
        with start_action(
            action_type="step",
            index=index,
            instructions=instructions,
            prompt=prompt,
            prompt_params=prompt_params,
            input_schema=input_schema,
            output_schema=output_schema,
        ):
            # Prepare: run pre-processing, and input validation.
            if output:
                instructions = None
                prompt = None
                msg_history = None
            else:
                instructions, prompt, msg_history = self.prepare(
                    index,
                    instructions,
                    prompt,
                    msg_history,
                    prompt_params,
                    api,
                    input_schema,
                    output_schema,
                )

            guard_logs.prompt = prompt
            guard_logs.instructions = instructions
            guard_logs.msg_history = msg_history

            # Call: run the API.
            llm_response = self.call(
                index, instructions, prompt, msg_history, api, output
            )

            guard_logs.llm_response = llm_response
            output = llm_response.output

            # Parse: parse the output.
            parsed_output, parsing_error = self.parse(index, output, output_schema)

            guard_logs.parsed_output = parsed_output

            # Validate: run output validation.
            validated_output = None
            if parsing_error and isinstance(parsed_output, NonParseableReAsk):
                reasks = self.introspect(index, parsed_output, output_schema)
            else:
                # Validate: run output validation.
                validated_output = self.validate(
                    guard_logs, index, parsed_output, output_schema
                )

                guard_logs.set_validated_output(
                    validated_output, self.full_schema_reask
                )

                # Introspect: inspect validated output for reasks.
                reasks = self.introspect(index, validated_output, output_schema)

            guard_logs.reasks = reasks

            # Replace reask values with fixed values if terminal step.
            if not self.do_loop(index, reasks):
                validated_output = sub_reasks_with_fixed_values(validated_output)

            guard_logs.set_validated_output(validated_output, self.full_schema_reask)

            return validated_output or parsed_output, reasks

    def prepare(
        self,
        index: int,
        instructions: Optional[Instructions],
        prompt: Prompt,
        msg_history: Optional[List[Dict]],
        prompt_params: Dict,
        api: Union[PromptCallableBase, AsyncPromptCallableBase],
        input_schema: Schema,
        output_schema: Schema,
    ) -> Tuple[Instructions, Prompt, List[Dict]]:
        """Prepare by running pre-processing and input validation.

        Returns:
            The instructions, prompt, and message history.
        """
        with start_action(action_type="prepare", index=index) as action:
            if prompt_params is None:
                prompt_params = {}

            if msg_history:
                msg_history = copy.deepcopy(msg_history)
                # Format any variables in the message history with the prompt params.
                for msg in msg_history:
                    msg["content"] = msg["content"].format(**prompt_params)

                prompt, instructions = None, None
            else:
                if isinstance(prompt, str):
                    prompt = Prompt(prompt)

                prompt = prompt.format(**prompt_params)

                # TODO(shreya): should there be any difference
                #  to parsing params for prompt?
                if instructions is not None and isinstance(instructions, Instructions):
                    instructions = instructions.format(**prompt_params)

                instructions, prompt = output_schema.preprocess_prompt(
                    api, instructions, prompt
                )

            action.log(
                message_type="info",
                instructions=instructions,
                prompt=prompt,
                prompt_params=prompt_params,
                validated_prompt_params=prompt_params,
            )

        return instructions, prompt, msg_history

    def call(
        self,
        index: int,
        instructions: Optional[Instructions],
        prompt: Prompt,
        msg_history: Optional[List[Dict[str, str]]],
        api: Callable,
        output: Optional[str] = None,
    ) -> LLMResponse:
        """Run a step.

        1. Query the LLM API,
        2. Convert the response string to a dict,
        3. Log the output
        """

        def msg_history_source(msg_history) -> List[Dict[str, str]]:
            msg_history_copy = copy.deepcopy(msg_history)
            for msg in msg_history_copy:
                msg["content"] = msg["content"].source
            return msg_history_copy

        with start_action(action_type="call", index=index, prompt=prompt) as action:
            llm_response = None
            try:
                if msg_history:
                    llm_response = api(
                        msg_history=msg_history_source(msg_history),
                        base_model=self.base_model,
                    )
                else:
                    if prompt and instructions:
                        llm_response = api(
                            prompt.source,
                            instructions=instructions.source,
                            base_model=self.base_model,
                        )
                    elif prompt:
                        llm_response = api(prompt.source, base_model=self.base_model)
            except Exception:
                # If the API call fails, try calling again without the base model.
                if msg_history:
                    llm_response = api(msg_history=msg_history_source(msg_history))
                else:
                    if prompt and instructions:
                        llm_response = api(
                            prompt.source, instructions=instructions.source
                        )
                    elif prompt:
                        llm_response = api(prompt.source)

            if llm_response is None:
                llm_response = LLMResponse(
                    output=output,
                )

            action.log(
                message_type="info",
                output=llm_response,
            )

            return llm_response

    def parse(
        self,
        index: int,
        output: str,
        output_schema: Schema,
    ):
        with start_action(action_type="parse", index=index) as action:
            parsed_output, error = output_schema.parse(output)

            action.log(
                message_type="info",
                parsed_output=parsed_output,
                error=error,
            )

            return parsed_output, error

    def validate(
        self,
        guard_logs: GuardLogs,
        index: int,
        parsed_output: Any,
        output_schema: Schema,
    ):
        """Validate the output."""
        with start_action(action_type="validate", index=index) as action:
            validated_output = output_schema.validate(
                guard_logs, parsed_output, self.metadata
            )

            action.log(
                message_type="info",
                validated_output=reasks_to_dict(validated_output),
            )

            return validated_output

    def introspect(
        self,
        index: int,
        validated_output: Any,
        output_schema: Schema,
    ) -> List[FieldReAsk]:
        """Introspect the validated output."""
        with start_action(action_type="introspect", index=index) as action:
            if validated_output is None:
                return []
            reasks = output_schema.introspect(validated_output)

            action.log(
                message_type="info",
                reasks=[r.__dict__ for r in reasks],
            )

            return reasks

    def do_loop(self, index: int, reasks: List[ReAsk]) -> bool:
        """Determine if we should loop again."""
        if reasks and index < self.num_reasks:
            return True
        return False

    def prepare_to_loop(
        self,
        reasks: list,
        validated_output: Optional[Dict],
        output_schema: Schema,
        include_instructions: bool = False,
        prompt_params: Dict = None,
    ) -> Tuple[Prompt, Instructions, Schema, Optional[List[Dict]]]:
        """Prepare to loop again."""
        output_schema, prompt, instructions = output_schema.get_reask_setup(
            reasks=reasks,
            original_response=validated_output,
            use_full_schema=self.full_schema_reask,
            prompt_params=prompt_params,
        )
        if not include_instructions:
            instructions = None
        msg_history = None  # clear msg history for reasking
        return prompt, instructions, output_schema, msg_history


class AsyncRunner(Runner):
    api: AsyncPromptCallableBase

    async def async_run(self, prompt_params: Dict = None) -> GuardHistory:
        """Execute the runner by repeatedly calling step until the reask budget
        is exhausted.

        Args:
            prompt_params: Parameters to pass to the prompt in order to
                generate the prompt string.

        Returns:
            The guard history.
        """
        self._reset_guard_history()

        # check if validator requirements are fulfilled
        missing_keys = verify_metadata_requirements(
            self.metadata, self.output_schema.to_dict().values()
        )
        if missing_keys:
            raise ValueError(
                f"Missing required metadata keys: {', '.join(missing_keys)}"
            )

        with start_action(
            action_type="run",
            instructions=self.instructions,
            prompt=self.prompt,
            api=self.api,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            num_reasks=self.num_reasks,
            metadata=self.metadata,
        ):
            instructions, prompt, msg_history, input_schema, output_schema = (
                self.instructions,
                self.prompt,
                self.msg_history,
                self.input_schema,
                self.output_schema,
            )
            for index in range(self.num_reasks + 1):
                # Run a single step.
                validated_output, reasks = await self.async_step(
                    index=index,
                    api=self.api,
                    instructions=instructions,
                    prompt=prompt,
                    msg_history=msg_history,
                    prompt_params=prompt_params,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    output=self.output if index == 0 else None,
                )

                # Loop again?
                if not self.do_loop(index, reasks):
                    break
                # Get new prompt and output schema.
                prompt, instructions, output_schema, msg_history = self.prepare_to_loop(
                    reasks,
                    validated_output,
                    output_schema,
                    prompt_params=prompt_params,
                )

            return self.guard_history

    async def async_step(
        self,
        index: int,
        api: AsyncPromptCallableBase,
        instructions: Optional[Instructions],
        prompt: Prompt,
        msg_history: Optional[List[Dict]],
        prompt_params: Dict,
        input_schema: Schema,
        output_schema: Schema,
        output: str = None,
    ):
        guard_logs = GuardLogs()
        self.guard_history.push(guard_logs)
        """Run a full step."""
        with start_action(
            action_type="step",
            index=index,
            instructions=instructions,
            prompt=prompt,
            prompt_params=prompt_params,
            input_schema=input_schema,
            output_schema=output_schema,
        ):
            # Prepare: run pre-processing, and input validation.
            if not output:
                instructions, prompt, msg_history = self.prepare(
                    index,
                    instructions,
                    prompt,
                    msg_history,
                    prompt_params,
                    api,
                    input_schema,
                    output_schema,
                )
            else:
                instructions = None
                prompt = None

            guard_logs.prompt = prompt
            guard_logs.instructions = instructions
            guard_logs.msg_history = msg_history

            # Call: run the API.
            llm_response = await self.async_call(
                index, instructions, prompt, msg_history, api, output
            )

            guard_logs.llm_response = llm_response
            output = llm_response.output

            # Parse: parse the output.
            parsed_output, parsing_error = self.parse(index, output, output_schema)

            guard_logs.parsed_output = parsed_output

            validated_output = None
            if parsing_error and isinstance(parsed_output, NonParseableReAsk):
                reasks = self.introspect(index, parsed_output, output_schema)
            else:
                # Validate: run output validation.
                validated_output = await self.async_validate(
                    guard_logs, index, parsed_output, output_schema
                )

                guard_logs.set_validated_output(
                    validated_output, self.full_schema_reask
                )

                # Introspect: inspect validated output for reasks.
                reasks = self.introspect(index, validated_output, output_schema)

            guard_logs.reasks = reasks

            # Replace reask values with fixed values if terminal step.
            if not self.do_loop(index, reasks):
                validated_output = sub_reasks_with_fixed_values(validated_output)

            guard_logs.set_validated_output(validated_output, self.full_schema_reask)

            return validated_output or parsed_output, reasks

    async def async_call(
        self,
        index: int,
        instructions: Optional[Instructions],
        prompt: Prompt,
        msg_history: Optional[List[Dict]],
        api: AsyncPromptCallableBase,
        output: Optional[str] = None,
    ) -> LLMResponse:
        """Run a step.

        1. Query the LLM API,
        2. Convert the response string to a dict,
        3. Log the output
        """
        with start_action(action_type="call", index=index, prompt=prompt) as action:
            llm_response = None
            try:
                if msg_history:
                    llm_response = await api(
                        msg_history=msg_history,
                        base_model=self.base_model,
                    )
                else:
                    if prompt and instructions:
                        llm_response = await api(
                            prompt.source,
                            instructions=instructions.source,
                            base_model=self.base_model,
                        )
                    elif prompt:
                        llm_response = await api(
                            prompt.source, base_model=self.base_model
                        )
            except Exception:
                # If the API call fails, try calling again without the base model.
                if msg_history:
                    llm_response = await api(msg_history=msg_history)
                else:
                    if prompt and instructions:
                        llm_response = await api(
                            prompt.source, instructions=instructions.source
                        )
                    elif prompt:
                        llm_response = await api(prompt.source)

            if llm_response is None:
                llm_response = LLMResponse(
                    output=output,
                )

            action.log(
                message_type="info",
                output=llm_response,
            )

            return llm_response

    async def async_validate(
        self,
        guard_logs: GuardLogs,
        index: int,
        parsed_output: Any,
        output_schema: Schema,
    ):
        """Validate the output."""
        with start_action(action_type="validate", index=index) as action:
            validated_output = await output_schema.async_validate(
                guard_logs, parsed_output, self.metadata
            )

            action.log(
                message_type="info",
                validated_output=reasks_to_dict(validated_output),
            )

            return validated_output
