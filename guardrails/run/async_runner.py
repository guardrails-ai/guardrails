import copy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel

from guardrails.classes.history import Call, Inputs, Iteration, Outputs
from guardrails.datatypes import verify_metadata_requirements
from guardrails.errors import ValidationError
from guardrails.llm_providers import AsyncPromptCallableBase, PromptCallableBase
from guardrails.prompt import Instructions, Prompt
from guardrails.run.runner import Runner
from guardrails.run.utils import msg_history_source, msg_history_string
from guardrails.schema import Schema, StringSchema
from guardrails.utils.exception_utils import UserFacingException
from guardrails.utils.llm_response import LLMResponse
from guardrails.utils.reask_utils import NonParseableReAsk, ReAsk
from guardrails.utils.telemetry_utils import async_trace


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
        base_model: Optional[
            Union[Type[BaseModel], Type[List[Type[BaseModel]]]]
        ] = None,
        full_schema_reask: bool = False,
        disable_tracer: Optional[bool] = True,
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
            disable_tracer=disable_tracer,
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
                    call_log.validation_response,
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

    @async_trace(name="step")
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
                iteration.outputs.validation_response = validated_output

                # Introspect: inspect validated output for reasks.
                reasks, valid_output = self.introspect(
                    index, validated_output, output_schema
                )
                iteration.outputs.guarded_output = valid_output

            iteration.outputs.reasks = reasks

        except Exception as e:
            error_message = str(e)
            iteration.outputs.error = error_message
            iteration.outputs.exception = e
            raise e
        return iteration

    @async_trace(name="call")
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
        # If the API supports a base model, pass it in.
        api_fn = api
        if api is not None:
            supports_base_model = getattr(api, "supports_base_model", False)
            if supports_base_model:
                api_fn = partial(api, base_model=self.base_model)

        if output is not None:
            llm_response = LLMResponse(
                output=output,
            )
        elif api_fn is None:
            raise ValueError("Either API or output must be provided.")
        elif msg_history:
            llm_response = await api_fn(msg_history=msg_history_source(msg_history))
        elif prompt and instructions:
            llm_response = await api_fn(prompt.source, instructions=instructions.source)
        elif prompt:
            llm_response = await api_fn(prompt.source)
        else:
            raise ValueError("'output', 'prompt' or 'msg_history' must be provided.")

        return llm_response

    async def async_validate(
        self,
        iteration: Iteration,
        index: int,
        parsed_output: Any,
        output_schema: Schema,
    ):
        """Validate the output."""
        validated_output = await output_schema.async_validate(
            iteration, parsed_output, self.metadata, attempt_number=index
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
                    raise ValidationError(
                        f"Message history validation failed: "
                        f"{validated_msg_history}"
                    )
                if validated_msg_history != msg_str:
                    raise ValidationError("Message history validation failed")
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
                iteration.outputs.validation_response = validated_prompt
                if validated_prompt is None:
                    raise ValidationError("Prompt validation failed")
                if isinstance(validated_prompt, ReAsk):
                    raise ValidationError(
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
                iteration.outputs.validation_response = validated_instructions
                if validated_instructions is None:
                    raise ValidationError("Instructions validation failed")
                if isinstance(validated_instructions, ReAsk):
                    raise ValidationError(
                        f"Instructions validation failed: {validated_instructions}"
                    )
                instructions = Instructions(validated_instructions)
        else:
            raise ValueError("Prompt or message history must be provided.")

        return instructions, prompt, msg_history
