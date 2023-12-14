import copy
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from eliot import add_destinations, start_action
from pydantic import BaseModel

from guardrails.classes.history import Call, Inputs, Iteration, Outputs
from guardrails.datatypes import verify_metadata_requirements
from guardrails.llm_providers import (
    AsyncPromptCallableBase,
    OpenAICallable,
    OpenAIChatCallable,
    PromptCallableBase,
)
from guardrails.logger import logger, set_scope
from guardrails.prompt import Instructions, Prompt
from guardrails.schema import Schema, StringSchema
from guardrails.utils.exception_utils import UserFacingException
from guardrails.utils.llm_response import LLMResponse
from guardrails.utils.openai_utils import OPENAI_VERSION
from guardrails.utils.reask_utils import (
    NonParseableReAsk,
    ReAsk,
    SkeletonReAsk,
    reasks_to_dict,
)
from guardrails.validator_base import ValidatorError

add_destinations(logger.debug)


class Runner:
    """Runner class that calls an LLM API with a prompt, and performs input and
    output validation.

    This class will repeatedly call the API until the
    reask budget is exhausted, or the output is valid.

    Args:
        prompt: The prompt to use.
        api: The LLM API to call, which should return a string.
        output_schema: The output schema to use for validation.
        num_reasks: The maximum number of times to reask the LLM in case of
            validation failure, defaults to 0.
        output: The output to use instead of calling the API, used in cases
            where the output is already known.
    """

    def __init__(
        self,
        output_schema: Schema,
        num_reasks: int,
        prompt: Optional[Union[str, Prompt]] = None,
        instructions: Optional[Union[str, Instructions]] = None,
        msg_history: Optional[List[Dict]] = None,
        api: Optional[PromptCallableBase] = None,
        prompt_schema: Optional[StringSchema] = None,
        instructions_schema: Optional[StringSchema] = None,
        msg_history_schema: Optional[StringSchema] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        base_model: Optional[Type[BaseModel]] = None,
        full_schema_reask: bool = False,
    ):
        if prompt:
            assert api, "Must provide an API if a prompt is provided."
            assert not output, "Cannot provide both a prompt and output."

        if isinstance(prompt, str):
            self.prompt = Prompt(prompt, output_schema=output_schema.transpile())
        else:
            self.prompt = prompt

        if isinstance(instructions, str):
            self.instructions = Instructions(
                instructions, output_schema=output_schema.transpile()
            )
        else:
            self.instructions = instructions

        if msg_history:
            msg_history = copy.deepcopy(msg_history)
            msg_history_copy = []
            for msg in msg_history:
                msg["content"] = Prompt(
                    msg["content"], output_schema=output_schema.transpile()
                )
                msg_history_copy.append(msg)
            self.msg_history = msg_history_copy
        else:
            self.msg_history = None

        self.api = api
        self.prompt_schema = prompt_schema
        self.instructions_schema = instructions_schema
        self.msg_history_schema = msg_history_schema
        self.output_schema = output_schema
        self.num_reasks = num_reasks
        self.metadata = metadata or {}
        self.output = output
        self.base_model = base_model
        self.full_schema_reask = full_schema_reask

    def __call__(self, call_log: Call, prompt_params: Optional[Dict] = None) -> Call:
        """Execute the runner by repeatedly calling step until the reask budget
        is exhausted.

        Args:
            prompt_params: Parameters to pass to the prompt in order to
                generate the prompt string.

        Returns:
            The Call log for this run.
        """
        try:
            if prompt_params is None:
                prompt_params = {}

            # check if validator requirements are fulfilled
            missing_keys = verify_metadata_requirements(
                self.metadata, self.output_schema.root_datatype
            )
            if missing_keys:
                raise ValueError(
                    f"Missing required metadata keys: {', '.join(missing_keys)}"
                )

            # Figure out if we need to include instructions in the prompt.
            include_instructions = not (
                self.instructions is None and self.msg_history is None
            )

            with start_action(
                action_type="run",
                instructions=self.instructions,
                prompt=self.prompt,
                api=self.api,
                prompt_schema=self.prompt_schema,
                instructions_schema=self.instructions_schema,
                msg_history_schema=self.msg_history_schema,
                output_schema=self.output_schema,
                num_reasks=self.num_reasks,
                metadata=self.metadata,
            ):
                (
                    instructions,
                    prompt,
                    msg_history,
                    prompt_schema,
                    instructions_schema,
                    msg_history_schema,
                    output_schema,
                ) = (
                    self.instructions,
                    self.prompt,
                    self.msg_history,
                    self.prompt_schema,
                    self.instructions_schema,
                    self.msg_history_schema,
                    self.output_schema,
                )
                for index in range(self.num_reasks + 1):
                    # Run a single step.
                    iteration = self.step(
                        index=index,
                        api=self.api,
                        instructions=instructions,
                        prompt=prompt,
                        msg_history=msg_history,
                        prompt_params=prompt_params,
                        prompt_schema=prompt_schema,
                        instructions_schema=instructions_schema,
                        msg_history_schema=msg_history_schema,
                        output_schema=output_schema,
                        output=self.output if index == 0 else None,
                        call_log=call_log,
                    )

                    # Loop again?
                    if not self.do_loop(index, iteration.reasks):
                        break

                    # Get new prompt and output schema.
                    (
                        prompt,
                        instructions,
                        output_schema,
                        msg_history,
                    ) = self.prepare_to_loop(
                        iteration.reasks,
                        call_log.validation_output,
                        output_schema,
                        prompt_params=prompt_params,
                        include_instructions=include_instructions,
                    )
        except UserFacingException as e:
            # Because Pydantic v1 doesn't respect property setters
            call_log._exception = e.original_exception
            raise e.original_exception
        except Exception as e:
            # Because Pydantic v1 doesn't respect property setters
            call_log._exception = e
            raise e
        return call_log

    def step(
        self,
        index: int,
        api: Optional[PromptCallableBase],
        instructions: Optional[Instructions],
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict]],
        prompt_params: Dict,
        prompt_schema: Optional[StringSchema],
        instructions_schema: Optional[StringSchema],
        msg_history_schema: Optional[StringSchema],
        output_schema: Schema,
        call_log: Call,
        output: Optional[str] = None,
    ) -> Iteration:
        """Run a full step."""
        inputs = Inputs(
            llm_api=api,
            llm_output=output,
            instructions=instructions,
            prompt=prompt,
            msg_history=msg_history,
            prompt_params=prompt_params,
            num_reasks=self.num_reasks,
            metadata=self.metadata,
            full_schema_reask=self.full_schema_reask,
        )
        outputs = Outputs()
        iteration = Iteration(inputs=inputs, outputs=outputs)
        set_scope(str(id(iteration)))
        call_log.iterations.push(iteration)

        try:
            with start_action(
                action_type="step",
                index=index,
                instructions=instructions,
                prompt=prompt,
                prompt_params=prompt_params,
                prompt_schema=prompt_schema,
                instructions_schema=instructions_schema,
                msg_history_schema=msg_history_schema,
                output_schema=output_schema,
            ):
                # Prepare: run pre-processing, and input validation.
                if output:
                    instructions = None
                    prompt = None
                    msg_history = None
                else:
                    instructions, prompt, msg_history = self.prepare(
                        call_log,
                        index,
                        instructions,
                        prompt,
                        msg_history,
                        prompt_params,
                        api,
                        prompt_schema,
                        instructions_schema,
                        msg_history_schema,
                        output_schema,
                    )

                iteration.inputs.instructions = instructions
                iteration.inputs.prompt = prompt
                iteration.inputs.msg_history = msg_history

                # Call: run the API.
                llm_response = self.call(
                    index, instructions, prompt, msg_history, api, output
                )

                iteration.outputs.llm_response_info = llm_response
                raw_output = llm_response.output

                # Parse: parse the output.
                parsed_output, parsing_error = self.parse(
                    index, raw_output, output_schema
                )
                if parsing_error:
                    iteration.outputs.exception = parsing_error
                    iteration.outputs.error = str(parsing_error)

                iteration.outputs.parsed_output = parsed_output

                # Validate: run output validation.
                if parsing_error and isinstance(parsed_output, NonParseableReAsk):
                    reasks, _ = self.introspect(index, parsed_output, output_schema)
                else:
                    # Validate: run output validation.
                    validated_output = self.validate(
                        iteration, index, parsed_output, output_schema
                    )
                    iteration.outputs.validation_output = validated_output

                    # Introspect: inspect validated output for reasks.
                    reasks, valid_output = self.introspect(
                        index, validated_output, output_schema
                    )
                    iteration.outputs.validated_output = valid_output

                iteration.outputs.reasks = reasks

        except Exception as e:
            error_message = str(e)
            iteration.outputs.error = error_message
            iteration.outputs.exception = e
            raise e
        return iteration

    def validate_msg_history(
        self,
        call_log: Call,
        msg_history: List[Dict],
        msg_history_schema: StringSchema,
    ):
        msg_str = msg_history_string(msg_history)
        inputs = Inputs(
            llm_output=msg_str,
        )
        iteration = Iteration(inputs=inputs)
        call_log.iterations.insert(0, iteration)
        validated_msg_history = msg_history_schema.validate(
            iteration, msg_str, self.metadata
        )
        iteration.outputs.validation_output = validated_msg_history
        if isinstance(validated_msg_history, ReAsk):
            raise ValidatorError(
                f"Message history validation failed: " f"{validated_msg_history}"
            )
        if validated_msg_history != msg_str:
            raise ValidatorError("Message history validation failed")

    def prepare_msg_history(
        self,
        call_log: Call,
        msg_history: List[Dict],
        prompt_params: Dict,
        msg_history_schema: Optional[StringSchema],
    ):
        msg_history = copy.deepcopy(msg_history)
        # Format any variables in the message history with the prompt params.
        for msg in msg_history:
            msg["content"] = msg["content"].format(**prompt_params)

        # validate msg_history
        if msg_history_schema is not None:
            self.validate_msg_history(call_log, msg_history, msg_history_schema)

        return msg_history

    def validate_prompt(
        self,
        call_log: Call,
        prompt_schema: StringSchema,
        prompt: Prompt,
    ):
        inputs = Inputs(
            llm_output=prompt.source,
        )
        iteration = Iteration(inputs=inputs)
        call_log.iterations.insert(0, iteration)
        validated_prompt = prompt_schema.validate(
            iteration, prompt.source, self.metadata
        )
        iteration.outputs.validation_output = validated_prompt
        if validated_prompt is None:
            raise ValidatorError("Prompt validation failed")
        if isinstance(validated_prompt, ReAsk):
            raise ValidatorError(f"Prompt validation failed: {validated_prompt}")
        return Prompt(validated_prompt)

    def validate_instructions(
        self,
        call_log: Call,
        instructions_schema: StringSchema,
        instructions: Instructions,
    ):
        inputs = Inputs(
            llm_output=instructions.source,
        )
        iteration = Iteration(inputs=inputs)
        call_log.iterations.insert(0, iteration)
        validated_instructions = instructions_schema.validate(
            iteration, instructions.source, self.metadata
        )
        iteration.outputs.validation_output = validated_instructions
        if validated_instructions is None:
            raise ValidatorError("Instructions validation failed")
        if isinstance(validated_instructions, ReAsk):
            raise ValidatorError(
                f"Instructions validation failed: {validated_instructions}"
            )
        return Instructions(validated_instructions)

    def prepare_prompt(
        self,
        call_log: Call,
        instructions: Optional[Instructions],
        prompt: Prompt,
        prompt_params: Dict,
        api: Union[PromptCallableBase, AsyncPromptCallableBase],
        prompt_schema: Optional[StringSchema],
        instructions_schema: Optional[StringSchema],
        output_schema: Schema,
    ):
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

        # validate prompt
        if prompt_schema is not None and prompt is not None:
            prompt = self.validate_prompt(call_log, prompt_schema, prompt)

        # validate instructions
        if instructions_schema is not None and instructions is not None:
            instructions = self.validate_instructions(
                call_log, instructions_schema, instructions
            )

        return instructions, prompt

    def prepare(
        self,
        call_log: Call,
        index: int,
        instructions: Optional[Instructions],
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict]],
        prompt_params: Dict,
        api: Optional[Union[PromptCallableBase, AsyncPromptCallableBase]],
        prompt_schema: Optional[StringSchema],
        instructions_schema: Optional[StringSchema],
        msg_history_schema: Optional[StringSchema],
        output_schema: Schema,
    ) -> Tuple[Optional[Instructions], Optional[Prompt], Optional[List[Dict]]]:
        """Prepare by running pre-processing and input validation.

        Returns:
            The instructions, prompt, and message history.
        """
        with start_action(action_type="prepare", index=index) as action:
            if api is None:
                raise UserFacingException(ValueError("API must be provided."))

            if prompt_params is None:
                prompt_params = {}

            if msg_history:
                if prompt_schema is not None or instructions_schema is not None:
                    raise UserFacingException(
                        ValueError(
                            "Prompt and instructions validation are "
                            "not supported when using message history."
                        )
                    )
                prompt, instructions = None, None
                msg_history = self.prepare_msg_history(
                    call_log, msg_history, prompt_params, msg_history_schema
                )
            elif prompt is not None:
                if msg_history_schema is not None:
                    raise UserFacingException(
                        ValueError(
                            "Message history validation is "
                            "not supported when using prompt/instructions."
                        )
                    )
                msg_history = None
                instructions, prompt = self.prepare_prompt(
                    call_log,
                    instructions,
                    prompt,
                    prompt_params,
                    api,
                    prompt_schema,
                    instructions_schema,
                    output_schema,
                )
            else:
                raise UserFacingException(
                    ValueError("'prompt' or 'msg_history' must be provided.")
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
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict[str, str]]],
        api: Optional[PromptCallableBase],
        output: Optional[str] = None,
    ) -> LLMResponse:
        """Run a step.

        1. Query the LLM API,
        2. Convert the response string to a dict,
        3. Log the output
        """

        with start_action(action_type="call", index=index, prompt=prompt) as action:
            if output is not None:
                llm_response = LLMResponse(
                    output=output,
                )
            elif api is None:
                raise ValueError("API or output must be provided.")
            elif msg_history:
                try:
                    llm_response = api(
                        msg_history=msg_history_source(msg_history),
                        base_model=self.base_model,
                    )
                except Exception:
                    # If the API call fails, try calling again without the base model.
                    llm_response = api(msg_history=msg_history_source(msg_history))
            elif prompt and instructions:
                try:
                    llm_response = api(
                        prompt.source,
                        instructions=instructions.source,
                        base_model=self.base_model,
                    )
                except Exception:
                    llm_response = api(prompt.source, instructions=instructions.source)
            elif prompt:
                try:
                    llm_response = api(prompt.source, base_model=self.base_model)
                except Exception:
                    llm_response = api(prompt.source)
            else:
                raise ValueError("'prompt' or 'msg_history' must be provided.")

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
        iteration: Iteration,
        index: int,
        parsed_output: Any,
        output_schema: Schema,
        **kwargs,
    ):
        """Validate the output."""
        with start_action(action_type="validate", index=index) as action:
            validated_output = output_schema.validate(
                iteration, parsed_output, self.metadata, **kwargs
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
    ) -> Tuple[Sequence[ReAsk], Optional[Union[str, Dict]]]:
        """Introspect the validated output."""
        with start_action(action_type="introspect", index=index) as action:
            if validated_output is None:
                return [], None
            reasks, valid_output = output_schema.introspect(validated_output)

            action.log(
                message_type="info",
                reasks=[r.__dict__ for r in reasks],
            )

            return reasks, valid_output

    def do_loop(self, index: int, reasks: Sequence[ReAsk]) -> bool:
        """Determine if we should loop again."""
        if reasks and index < self.num_reasks:
            return True
        return False

    def prepare_to_loop(
        self,
        reasks: Sequence[ReAsk],
        validated_output: Optional[Union[str, Dict, ReAsk]],
        output_schema: Schema,
        prompt_params: Dict,
        include_instructions: bool = False,
    ) -> Tuple[Prompt, Optional[Instructions], Schema, Optional[List[Dict]]]:
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
    def __init__(
        self,
        output_schema: Schema,
        num_reasks: int,
        prompt: Optional[Union[str, Prompt]] = None,
        instructions: Optional[Union[str, Instructions]] = None,
        msg_history: Optional[List[Dict]] = None,
        api: Optional[AsyncPromptCallableBase] = None,
        prompt_schema: Optional[StringSchema] = None,
        instructions_schema: Optional[StringSchema] = None,
        msg_history_schema: Optional[StringSchema] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        base_model: Optional[Type[BaseModel]] = None,
        full_schema_reask: bool = False,
    ):
        super().__init__(
            output_schema=output_schema,
            num_reasks=num_reasks,
            prompt=prompt,
            instructions=instructions,
            msg_history=msg_history,
            api=api,
            prompt_schema=prompt_schema,
            instructions_schema=instructions_schema,
            msg_history_schema=msg_history_schema,
            metadata=metadata,
            output=output,
            base_model=base_model,
            full_schema_reask=full_schema_reask,
        )
        self.api: Optional[AsyncPromptCallableBase] = api

    async def async_run(
        self, call_log: Call, prompt_params: Optional[Dict] = None
    ) -> Call:
        """Execute the runner by repeatedly calling step until the reask budget
        is exhausted.

        Args:
            prompt_params: Parameters to pass to the prompt in order to
                generate the prompt string.

        Returns:
            The Call log for this run.
        """
        try:
            if prompt_params is None:
                prompt_params = {}

            # check if validator requirements are fulfilled
            missing_keys = verify_metadata_requirements(
                self.metadata, self.output_schema.root_datatype
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
                prompt_schema=self.prompt_schema,
                instructions_schema=self.instructions_schema,
                msg_history_schema=self.msg_history_schema,
                output_schema=self.output_schema,
                num_reasks=self.num_reasks,
                metadata=self.metadata,
            ):
                (
                    instructions,
                    prompt,
                    msg_history,
                    prompt_schema,
                    instructions_schema,
                    msg_history_schema,
                    output_schema,
                ) = (
                    self.instructions,
                    self.prompt,
                    self.msg_history,
                    self.prompt_schema,
                    self.instructions_schema,
                    self.msg_history_schema,
                    self.output_schema,
                )
                for index in range(self.num_reasks + 1):
                    # Run a single step.
                    iteration = await self.async_step(
                        index=index,
                        api=self.api,
                        instructions=instructions,
                        prompt=prompt,
                        msg_history=msg_history,
                        prompt_params=prompt_params,
                        prompt_schema=prompt_schema,
                        instructions_schema=instructions_schema,
                        msg_history_schema=msg_history_schema,
                        output_schema=output_schema,
                        output=self.output if index == 0 else None,
                        call_log=call_log,
                    )

                    # Loop again?
                    if not self.do_loop(index, iteration.reasks):
                        break

                    # Get new prompt and output schema.
                    (
                        prompt,
                        instructions,
                        output_schema,
                        msg_history,
                    ) = self.prepare_to_loop(
                        iteration.reasks,
                        call_log.validation_output,
                        output_schema,
                        prompt_params=prompt_params,
                    )
        except UserFacingException as e:
            # Because Pydantic v1 doesn't respect property setters
            call_log._exception = e.original_exception
            raise e.original_exception
        except Exception as e:
            # Because Pydantic v1 doesn't respect property setters
            call_log._exception = e
            raise e

        return call_log

    async def async_step(
        self,
        index: int,
        api: Optional[AsyncPromptCallableBase],
        instructions: Optional[Instructions],
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict]],
        prompt_params: Dict,
        prompt_schema: Optional[StringSchema],
        instructions_schema: Optional[StringSchema],
        msg_history_schema: Optional[StringSchema],
        output_schema: Schema,
        call_log: Call,
        output: Optional[str] = None,
    ) -> Iteration:
        """Run a full step."""
        inputs = Inputs(
            llm_api=api,
            llm_output=output,
            instructions=instructions,
            prompt=prompt,
            msg_history=msg_history,
            prompt_params=prompt_params,
            num_reasks=self.num_reasks,
            metadata=self.metadata,
            full_schema_reask=self.full_schema_reask,
        )
        outputs = Outputs()
        iteration = Iteration(inputs=inputs, outputs=outputs)
        call_log.iterations.push(iteration)

        try:
            with start_action(
                action_type="step",
                index=index,
                instructions=instructions,
                prompt=prompt,
                prompt_params=prompt_params,
                prompt_schema=prompt_schema,
                instructions_schema=instructions_schema,
                msg_history_schema=msg_history_schema,
                output_schema=output_schema,
            ):
                # Prepare: run pre-processing, and input validation.
                if output:
                    instructions = None
                    prompt = None
                    msg_history = None
                else:
                    instructions, prompt, msg_history = await self.async_prepare(
                        call_log,
                        index,
                        instructions,
                        prompt,
                        msg_history,
                        prompt_params,
                        api,
                        prompt_schema,
                        instructions_schema,
                        msg_history_schema,
                        output_schema,
                    )

                iteration.inputs.instructions = instructions
                iteration.inputs.prompt = prompt
                iteration.inputs.msg_history = msg_history

                # Call: run the API.
                llm_response = await self.async_call(
                    index, instructions, prompt, msg_history, api, output
                )

                iteration.outputs.llm_response_info = llm_response
                output = llm_response.output

                # Parse: parse the output.
                parsed_output, parsing_error = self.parse(index, output, output_schema)
                if parsing_error:
                    # Parsing errors are captured and not raised
                    #   because they are recoverable
                    #   i.e. result in a reask
                    iteration.outputs.exception = parsing_error
                    iteration.outputs.error = str(parsing_error)

                iteration.outputs.parsed_output = parsed_output

                if parsing_error and isinstance(parsed_output, NonParseableReAsk):
                    reasks, _ = self.introspect(index, parsed_output, output_schema)
                else:
                    # Validate: run output validation.
                    validated_output = await self.async_validate(
                        iteration, index, parsed_output, output_schema
                    )
                    iteration.outputs.validation_output = validated_output

                    # Introspect: inspect validated output for reasks.
                    reasks, valid_output = self.introspect(
                        index, validated_output, output_schema
                    )
                    iteration.outputs.validated_output = valid_output

                iteration.outputs.reasks = reasks

        except Exception as e:
            error_message = str(e)
            iteration.outputs.error = error_message
            iteration.outputs.exception = e
            raise e
        return iteration

    async def async_call(
        self,
        index: int,
        instructions: Optional[Instructions],
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict]],
        api: Optional[AsyncPromptCallableBase],
        output: Optional[str] = None,
    ) -> LLMResponse:
        """Run a step.

        1. Query the LLM API,
        2. Convert the response string to a dict,
        3. Log the output
        """
        with start_action(action_type="call", index=index, prompt=prompt) as action:
            if output is not None:
                llm_response = LLMResponse(
                    output=output,
                )
            elif api is None:
                raise ValueError("Either API or output must be provided.")
            elif msg_history:
                try:
                    llm_response = await api(
                        msg_history=msg_history_source(msg_history),
                        base_model=self.base_model,
                    )
                except Exception:
                    # If the API call fails, try calling again without the base model.
                    llm_response = await api(
                        msg_history=msg_history_source(msg_history)
                    )
            elif prompt and instructions:
                try:
                    llm_response = await api(
                        prompt.source,
                        instructions=instructions.source,
                        base_model=self.base_model,
                    )
                except Exception:
                    llm_response = await api(
                        prompt.source, instructions=instructions.source
                    )
            elif prompt:
                try:
                    llm_response = await api(prompt.source, base_model=self.base_model)
                except Exception:
                    llm_response = await api(prompt.source)
            else:
                raise ValueError(
                    "'output', 'prompt' or 'msg_history' must be provided."
                )

            action.log(
                message_type="info",
                output=llm_response,
            )

            return llm_response

    async def async_validate(
        self,
        iteration: Iteration,
        index: int,
        parsed_output: Any,
        output_schema: Schema,
    ):
        """Validate the output."""
        with start_action(action_type="validate", index=index) as action:
            validated_output = await output_schema.async_validate(
                iteration, parsed_output, self.metadata
            )

            action.log(
                message_type="info",
                validated_output=reasks_to_dict(validated_output),
            )

            return validated_output

    async def async_prepare(
        self,
        call_log: Call,
        index: int,
        instructions: Optional[Instructions],
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict]],
        prompt_params: Dict,
        api: Optional[Union[PromptCallableBase, AsyncPromptCallableBase]],
        prompt_schema: Optional[StringSchema],
        instructions_schema: Optional[StringSchema],
        msg_history_schema: Optional[StringSchema],
        output_schema: Schema,
    ) -> Tuple[Optional[Instructions], Optional[Prompt], Optional[List[Dict]]]:
        """Prepare by running pre-processing and input validation.

        Returns:
            The instructions, prompt, and message history.
        """
        with start_action(action_type="prepare", index=index) as action:
            if api is None:
                raise ValueError("API must be provided.")

            if prompt_params is None:
                prompt_params = {}

            if msg_history:
                msg_history = copy.deepcopy(msg_history)
                # Format any variables in the message history with the prompt params.
                for msg in msg_history:
                    msg["content"] = msg["content"].format(**prompt_params)

                prompt, instructions = None, None

                # validate msg_history
                if msg_history_schema is not None:
                    msg_str = msg_history_string(msg_history)
                    inputs = Inputs(
                        llm_output=msg_str,
                    )
                    iteration = Iteration(inputs=inputs)
                    call_log.iterations.insert(0, iteration)
                    validated_msg_history = await msg_history_schema.async_validate(
                        iteration, msg_str, self.metadata
                    )
                    if isinstance(validated_msg_history, ReAsk):
                        raise ValidatorError(
                            f"Message history validation failed: "
                            f"{validated_msg_history}"
                        )
                    if validated_msg_history != msg_str:
                        raise ValidatorError("Message history validation failed")
            elif prompt is not None:
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

                # validate prompt
                if prompt_schema is not None and prompt is not None:
                    inputs = Inputs(
                        llm_output=prompt.source,
                    )
                    iteration = Iteration(inputs=inputs)
                    call_log.iterations.insert(0, iteration)
                    validated_prompt = await prompt_schema.async_validate(
                        iteration, prompt.source, self.metadata
                    )
                    iteration.outputs.validation_output = validated_prompt
                    if validated_prompt is None:
                        raise ValidatorError("Prompt validation failed")
                    if isinstance(validated_prompt, ReAsk):
                        raise ValidatorError(
                            f"Prompt validation failed: {validated_prompt}"
                        )
                    prompt = Prompt(validated_prompt)

                # validate instructions
                if instructions_schema is not None and instructions is not None:
                    inputs = Inputs(
                        llm_output=instructions.source,
                    )
                    iteration = Iteration(inputs=inputs)
                    call_log.iterations.insert(0, iteration)
                    validated_instructions = await instructions_schema.async_validate(
                        iteration, instructions.source, self.metadata
                    )
                    iteration.outputs.validation_output = validated_instructions
                    if validated_instructions is None:
                        raise ValidatorError("Instructions validation failed")
                    if isinstance(validated_instructions, ReAsk):
                        raise ValidatorError(
                            f"Instructions validation failed: {validated_instructions}"
                        )
                    instructions = Instructions(validated_instructions)
            else:
                raise ValueError("Prompt or message history must be provided.")

            action.log(
                message_type="info",
                instructions=instructions,
                prompt=prompt,
                prompt_params=prompt_params,
                validated_prompt_params=prompt_params,
            )

        return instructions, prompt, msg_history


class StreamRunner(Runner):
    """Runner class that calls a streaming LLM API with a prompt.

    This class performs output validation when the output is a stream of
    chunks. Inherits from Runner class, as overall structure remains
    similar.
    """

    def __call__(self, call_log: Call, prompt_params: Optional[Dict] = None):
        """Execute the StreamRunner.

        Args:
            prompt_params: Parameters to pass to the prompt in order to
                generate the prompt string.

        Returns:
            The Call log for this run.
        """
        if prompt_params is None:
            prompt_params = {}

        # check if validator requirements are fulfilled
        missing_keys = verify_metadata_requirements(
            self.metadata, self.output_schema.root_datatype
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
            prompt_schema=self.prompt_schema,
            instructions_schema=self.instructions_schema,
            msg_history_schema=self.msg_history_schema,
            output_schema=self.output_schema,
            num_reasks=self.num_reasks,
            metadata=self.metadata,
        ):
            (
                instructions,
                prompt,
                msg_history,
                prompt_schema,
                instructions_schema,
                msg_history_schema,
                output_schema,
            ) = (
                self.instructions,
                self.prompt,
                self.msg_history,
                self.prompt_schema,
                self.instructions_schema,
                self.msg_history_schema,
                self.output_schema,
            )

            return self.step(
                index=0,
                api=self.api,
                instructions=instructions,
                prompt=prompt,
                msg_history=msg_history,
                prompt_params=prompt_params,
                prompt_schema=prompt_schema,
                instructions_schema=instructions_schema,
                msg_history_schema=msg_history_schema,
                output_schema=output_schema,
                output=self.output,
                call_log=call_log,
            )

    def step(
        self,
        index: int,
        api: Optional[PromptCallableBase],
        instructions: Optional[Instructions],
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict]],
        prompt_params: Dict,
        prompt_schema: Optional[StringSchema],
        instructions_schema: Optional[StringSchema],
        msg_history_schema: Optional[StringSchema],
        output_schema: Schema,
        call_log: Call,
        output: Optional[str] = None,
    ):
        """Run a full step."""
        inputs = Inputs(
            llm_api=api,
            llm_output=output,
            instructions=instructions,
            prompt=prompt,
            msg_history=msg_history,
            prompt_params=prompt_params,
            num_reasks=self.num_reasks,
            metadata=self.metadata,
            full_schema_reask=self.full_schema_reask,
        )
        outputs = Outputs()
        iteration = Iteration(inputs=inputs, outputs=outputs)
        call_log.iterations.push(iteration)

        with start_action(
            action_type="step",
            index=index,
            instructions=instructions,
            prompt=prompt,
            prompt_params=prompt_params,
            prompt_schema=prompt_schema,
            instructions_schema=instructions_schema,
            msg_history_schema=msg_history_schema,
            output_schema=output_schema,
        ):
            # Prepare: run pre-processing, and input validation.
            if output:
                instructions = None
                prompt = None
                msg_history = None
            else:
                instructions, prompt, msg_history = self.prepare(
                    call_log,
                    index,
                    instructions,
                    prompt,
                    msg_history,
                    prompt_params,
                    api,
                    prompt_schema,
                    instructions_schema,
                    msg_history_schema,
                    output_schema,
                )

            iteration.inputs.prompt = prompt
            iteration.inputs.instructions = instructions
            iteration.inputs.msg_history = msg_history

            # Call: run the API that returns a generator wrapped in LLMResponse
            llm_response = self.call(
                index, instructions, prompt, msg_history, api, output
            )

            # Get the stream (generator) from the LLMResponse
            stream = llm_response.stream_output
            if stream is None:
                raise ValueError(
                    "No stream was returned from the API. Please check that "
                    "the API is returning a generator."
                )

            fragment = ""
            parsed_fragment, validated_fragment, valid_op = None, None, None
            verified = set()
            # Loop over the stream
            # and construct "fragments" of concatenated chunks
            for chunk in stream:
                # 1. Get the text from the chunk and append to fragment
                chunk_text = self.get_chunk_text(chunk, api)
                fragment += chunk_text

                # 2. Parse the fragment
                parsed_fragment, move_to_next = self.parse(
                    index, fragment, output_schema, verified
                )
                if move_to_next:
                    # Continue to next chunk
                    continue

                # 3. Run output validation
                validated_fragment = self.validate(
                    iteration,
                    index,
                    parsed_fragment,
                    output_schema,
                    validate_subschema=True,
                )
                if isinstance(validated_fragment, SkeletonReAsk):
                    raise ValueError(
                        "Received fragment schema is an invalid sub-schema "
                        "of the expected output JSON schema."
                    )

                # 4. Introspect: inspect the validated fragment for reasks
                reasks, valid_op = self.introspect(
                    index, validated_fragment, output_schema
                )
                if reasks:
                    raise ValueError(
                        "Reasks are not yet supported with streaming. Please "
                        "remove reasks from schema or disable streaming."
                    )

                # 5. Convert validated fragment to a pretty JSON string
                try:
                    pretty_validated_fragment = json.dumps(valid_op, indent=4)
                except Exception as e:
                    raise ValueError(
                        f"Error formatting validated fragment JSON: {e}"
                    ) from e

                # 6. Yield raw and validated fragments
                raw_yield = f"Raw LLM response:\n{fragment}\n"
                validated_yield = (
                    f"\nValidated response:\n{pretty_validated_fragment}\n"
                )

                yield raw_yield + validated_yield

        # Finally, add to logs
        iteration.outputs.raw_output = fragment
        iteration.outputs.parsed_output = parsed_fragment
        iteration.outputs.validation_output = validated_fragment
        iteration.outputs.validated_output = valid_op

    def get_chunk_text(self, chunk: Any, api: Union[PromptCallableBase, None]) -> str:
        """Get the text from a chunk."""
        chunk_text = ""
        if isinstance(api, OpenAICallable):
            if OPENAI_VERSION.startswith("0"):
                finished = chunk["choices"][0]["finish_reason"]
                if "text" in chunk["choices"][0]:
                    content = chunk["choices"][0]["text"]
                    if not finished and content:
                        chunk_text = content
            else:
                finished = chunk.choices[0].finish_reason
                content = chunk.choices[0].text
                if not finished and content:
                    chunk_text = content
        elif isinstance(api, OpenAIChatCallable):
            if OPENAI_VERSION.startswith("0"):
                finished = chunk["choices"][0]["finish_reason"]
                if "content" in chunk["choices"][0]["delta"]:
                    content = chunk["choices"][0]["delta"]["content"]
                    if not finished and content:
                        chunk_text = content
            else:
                finished = chunk.choices[0].finish_reason
                content = chunk.choices[0].delta.content
                if not finished and content:
                    chunk_text = content
        else:
            try:
                chunk_text = chunk
            except Exception as e:
                raise ValueError(
                    f"Error getting chunk from stream: {e}. "
                    "Non-OpenAI API callables expected to return "
                    "a generator of strings."
                ) from e
        return chunk_text

    def parse(
        self,
        index: int,
        output: str,
        output_schema: Schema,
        verified: set,
    ):
        """Parse the output."""
        with start_action(action_type="parse", index=index) as action:
            parsed_output, error = output_schema.parse(
                output, stream=True, verified=verified
            )

            # Error can be either of (True/False/None/string representing error)
            if error:
                # If parsing error is a string,
                # it is an error from output_schema.parse_fragment()
                if isinstance(error, str):
                    raise ValueError("Unable to parse output: " + error)
            # Else if either of (None/True/False), return parsed_output and error

            action.log(
                message_type="info",
                parsed_output=parsed_output,
                error=error,
            )

            return parsed_output, error


def msg_history_source(msg_history) -> List[Dict[str, str]]:
    msg_history_copy = copy.deepcopy(msg_history)
    for msg in msg_history_copy:
        msg["content"] = msg["content"].source
    return msg_history_copy


def msg_history_string(msg_history) -> str:
    msg_history_copy = ""
    for msg in msg_history:
        msg_history_copy += msg["content"].source
    return msg_history_copy
