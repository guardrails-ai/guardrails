import copy
from functools import partial
from typing import Any, Dict, List, Optional, cast


from guardrails import validator_service
from guardrails.classes.execution.guard_execution_options import GuardExecutionOptions
from guardrails.classes.history import Call, Inputs, Iteration, Outputs
from guardrails.classes.output_type import OutputTypes
from guardrails.errors import ValidationError
from guardrails.llm_providers import AsyncPromptCallableBase
from guardrails.logger import set_scope
from guardrails.run.runner import Runner
from guardrails.run.utils import messages_source
from guardrails.schema.validator import schema_validation
from guardrails.hub_telemetry.hub_tracing import async_trace
from guardrails.types.inputs import MessageHistory
from guardrails.types.pydantic import ModelOrListOfModels
from guardrails.types.validator import ValidatorMap
from guardrails.utils.exception_utils import UserFacingException
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.actions.reask import NonParseableReAsk, ReAsk
from guardrails.telemetry import trace_async_call, trace_async_step

from guardrails.constants import fail_status
from guardrails.prompt import Prompt


class AsyncRunner(Runner):
    def __init__(
        self,
        output_type: OutputTypes,
        output_schema: Dict[str, Any],
        num_reasks: int,
        validation_map: ValidatorMap,
        *,
        messages: Optional[List[Dict]] = None,
        api: Optional[AsyncPromptCallableBase] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        base_model: Optional[ModelOrListOfModels] = None,
        full_schema_reask: bool = False,
        disable_tracer: Optional[bool] = True,
        exec_options: Optional[GuardExecutionOptions] = None,
    ):
        super().__init__(
            output_type=output_type,
            output_schema=output_schema,
            num_reasks=num_reasks,
            validation_map=validation_map,
            messages=messages,
            api=api,
            metadata=metadata,
            output=output,
            base_model=base_model,
            full_schema_reask=full_schema_reask,
            disable_tracer=disable_tracer,
            exec_options=exec_options,
        )
        self.api = api

    # TODO: Refactor this to use inheritance and overrides
    # Why are we using a different method here instead of just overriding?
    @async_trace(name="/reasks", origin="AsyncRunner.async_run")
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
        prompt_params = prompt_params or {}
        try:
            (
                messages,
                output_schema,
            ) = (
                self.messages,
                self.output_schema,
            )
            index = 0
            for index in range(self.num_reasks + 1):
                # Run a single step.
                iteration = await self.async_step(
                    index=index,
                    api=self.api,
                    messages=messages,
                    prompt_params=prompt_params,
                    output_schema=output_schema,
                    output=self.output if index == 0 else None,
                    call_log=call_log,
                )

                # Loop again?
                if not self.do_loop(index, iteration.reasks):
                    break

                # Get new prompt and output schema.
                (
                    output_schema,
                    messages,
                ) = self.prepare_to_loop(
                    iteration.reasks,
                    output_schema,
                    parsed_output=iteration.outputs.parsed_output,
                    validated_output=call_log.validation_response,
                    prompt_params=prompt_params,
                )

        except UserFacingException as e:
            # Because Pydantic v1 doesn't respect property setters
            call_log.exception = e.original_exception
            raise e.original_exception
        except Exception as e:
            # Because Pydantic v1 doesn't respect property setters
            call_log.exception = e
            raise e

        return call_log

    # TODO: Refactor this to use inheritance and overrides
    @async_trace(name="/step", origin="AsyncRunner.async_step")
    @trace_async_step
    async def async_step(
        self,
        index: int,
        output_schema: Dict[str, Any],
        call_log: Call,
        *,
        api: Optional[AsyncPromptCallableBase],
        messages: Optional[List[Dict]] = None,
        prompt_params: Optional[Dict] = None,
        output: Optional[str] = None,
    ) -> Iteration:
        """Run a full step."""
        prompt_params = prompt_params or {}
        inputs = Inputs(
            llm_api=api,
            llm_output=output,
            messages=messages,
            prompt_params=prompt_params,
            num_reasks=self.num_reasks,
            metadata=self.metadata,
            full_schema_reask=self.full_schema_reask,
        )
        outputs = Outputs()
        iteration = Iteration(
            call_id=call_log.id, index=index, inputs=inputs, outputs=outputs
        )
        set_scope(str(id(iteration)))
        call_log.iterations.push(iteration)

        try:
            # Prepare: run pre-processing, and input validation.
            if output is not None:
                messages = None
            else:
                messages = await self.async_prepare(
                    call_log,
                    messages=messages,
                    prompt_params=prompt_params,
                    api=api,
                    attempt_number=index,
                )

            iteration.inputs.messages = messages

            # Call: run the API.
            llm_response = await self.async_call(messages, api, output)

            iteration.outputs.llm_response_info = llm_response
            output = llm_response.output

            # Parse: parse the output.
            parsed_output, parsing_error = self.parse(output, output_schema)
            if parsing_error:
                # Parsing errors are captured and not raised
                #   because they are recoverable
                #   i.e. result in a reask
                iteration.outputs.exception = parsing_error  # type: ignore  # pyright and pydantic don't agree
                iteration.outputs.error = str(parsing_error)

            iteration.outputs.parsed_output = parsed_output  # type: ignore  # pyright and pydantic don't agree

            if parsing_error and isinstance(parsed_output, NonParseableReAsk):
                reasks, _ = self.introspect(parsed_output)
            else:
                # Validate: run output validation.
                validated_output = await self.async_validate(
                    iteration, index, parsed_output, output_schema
                )
                iteration.outputs.validation_response = validated_output

                # Introspect: inspect validated output for reasks.
                reasks, valid_output = self.introspect(validated_output)
                iteration.outputs.guarded_output = valid_output

            iteration.outputs.reasks = reasks  # type: ignore  # pyright and pydantic don't agree

        except Exception as e:
            error_message = str(e)
            iteration.outputs.error = error_message
            iteration.outputs.exception = e
            raise e
        return iteration

    # TODO: Refactor this to use inheritance and overrides
    @async_trace(name="/llm_call", origin="AsyncRunner.async_call")
    @trace_async_call
    async def async_call(
        self,
        messages: Optional[List[Dict]],
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
            raise ValueError("API or output must be provided.")
        elif messages:
            llm_response = await api_fn(messages=messages_source(messages))
        else:
            llm_response = await api_fn()
        return llm_response

    # TODO: Refactor this to use inheritance and overrides
    @async_trace(name="/validation", origin="AsyncRunner.async_validate")
    async def async_validate(
        self,
        iteration: Iteration,
        attempt_number: int,
        parsed_output: Any,
        output_schema: Dict[str, Any],
        stream: Optional[bool] = False,
        **kwargs,
    ):
        """Validate the output."""
        # Break early if empty
        if parsed_output is None:
            return None

        skeleton_reask = schema_validation(parsed_output, output_schema, **kwargs)
        if skeleton_reask:
            return skeleton_reask

        if self.output_type != OutputTypes.STRING:
            stream = None

        validated_output, _metadata = await validator_service.async_validate(
            value=parsed_output,
            metadata=self.metadata,
            validator_map=self.validation_map,
            iteration=iteration,
            disable_tracer=self._disable_tracer,
            path="$",
            stream=stream,
            **kwargs,
        )
        validated_output = validator_service.post_process_validation(
            validated_output, attempt_number, iteration, self.output_type
        )

        return validated_output

    # TODO: Refactor this to use inheritance and overrides
    @async_trace(name="/input_prep", origin="AsyncRunner.async_prepare")
    async def async_prepare(
        self,
        call_log: Call,
        attempt_number: int,
        *,
        messages: Optional[List[Dict]],
        prompt_params: Optional[Dict] = None,
        api: Optional[AsyncPromptCallableBase],
    ) -> Optional[List[Dict]]:
        """Prepare by running pre-processing and input validation.

        Returns:
            The messages.
        """
        prompt_params = prompt_params or {}
        if api is None:
            raise UserFacingException(ValueError("API must be provided."))

        if messages:
            # Runner.prepare_messages
            messages = await self.prepare_messages(
                call_log=call_log,
                messages=messages,
                prompt_params=prompt_params,
                attempt_number=attempt_number,
            )

        else:
            raise UserFacingException(ValueError("'messages' must be provided."))

        return messages

    async def prepare_messages(
        self,
        call_log: Call,
        messages: MessageHistory,
        prompt_params: Dict,
        attempt_number: int,
    ) -> MessageHistory:
        formatted_messages = []

        # Format any variables in the message history with the prompt params.
        for msg in messages:
            msg_copy = copy.deepcopy(msg)
            msg_copy["content"] = msg_copy["content"].format(**prompt_params)
            formatted_messages.append(msg_copy)

        if "messages" in self.validation_map:
            await self.validate_messages(call_log, formatted_messages, attempt_number)

        return formatted_messages

    @async_trace(name="/input_validation", origin="AsyncRunner.validate_messages")
    async def validate_messages(
        self, call_log: Call, messages: MessageHistory, attempt_number: int
    ):
        for msg in messages:
            content = (
                msg["content"].source
                if isinstance(msg["content"], Prompt)
                else msg["content"]
            )
            inputs = Inputs(
                llm_output=content,
            )
            iteration = Iteration(
                call_id=call_log.id, index=attempt_number, inputs=inputs
            )
            call_log.iterations.insert(0, iteration)
            value, _metadata = await validator_service.async_validate(
                value=content,
                metadata=self.metadata,
                validator_map=self.validation_map,
                iteration=iteration,
                disable_tracer=self._disable_tracer,
                path="messages",
            )

            validated_msg = validator_service.post_process_validation(
                value, attempt_number, iteration, OutputTypes.STRING
            )

            iteration.outputs.validation_response = validated_msg

            if isinstance(validated_msg, ReAsk):
                raise ValidationError(f"Messages validation failed: {validated_msg}")
            elif not validated_msg or iteration.status == fail_status:
                raise ValidationError("Messages validation failed")

            msg["content"] = cast(str, validated_msg)

        return messages  # type: ignore
