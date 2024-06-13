import copy
from collections.abc import Awaitable
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union, cast


from guardrails import validator_service
from guardrails.classes.execution.guard_execution_options import GuardExecutionOptions
from guardrails.classes.history import Call, Inputs, Iteration, Outputs
from guardrails.classes.output_type import OutputTypes
from guardrails.constants import fail_status
from guardrails.errors import ValidationError
from guardrails.llm_providers import AsyncPromptCallableBase, PromptCallableBase
from guardrails.logger import set_scope
from guardrails.run.runner import Runner
from guardrails.run.utils import messages_source, messages_string
from guardrails.schema.validator import schema_validation
from guardrails.types.pydantic import ModelOrListOfModels
from guardrails.types.validator import ValidatorMap
from guardrails.utils.exception_utils import UserFacingException
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.utils.prompt_utils import preprocess_prompt, prompt_uses_xml
from guardrails.actions.reask import NonParseableReAsk, ReAsk
from guardrails.utils.telemetry_utils import async_trace


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
        self.api: Optional[AsyncPromptCallableBase] = api

    # TODO: Refactor this to use inheritance and overrides
    # Why are we using a different method here instead of just overriding?
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

            # Log how many times we reasked
            # Use the HubTelemetry singleton
            if not self._disable_tracer:
                self._hub_telemetry.create_new_span(
                    span_name="/reasks",
                    attributes=[("reask_count", index)],
                    is_parent=False,  # This span has no children
                    has_parent=True,  # This span has a parent
                )
        except UserFacingException as e:
            # Because Pydantic v1 doesn't respect property setters
            call_log._set_exception(e.original_exception)
            raise e.original_exception
        except Exception as e:
            # Because Pydantic v1 doesn't respect property setters
            call_log._set_exception(e)
            raise e

        return call_log

    # TODO: Refactor this to use inheritance and overrides
    @async_trace(name="step")
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
        iteration = Iteration(inputs=inputs, outputs=outputs)
        set_scope(str(id(iteration)))
        call_log.iterations.push(iteration)

        try:
            # Prepare: run pre-processing, and input validation.
            if output:
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
            llm_response = await self.async_call(
                messages, api, output
            )

            iteration.outputs.llm_response_info = llm_response
            output = llm_response.output

            # Parse: parse the output.
            parsed_output, parsing_error = self.parse(output, output_schema)
            if parsing_error:
                # Parsing errors are captured and not raised
                #   because they are recoverable
                #   i.e. result in a reask
                iteration.outputs.exception = parsing_error
                iteration.outputs.error = str(parsing_error)

            iteration.outputs.parsed_output = parsed_output

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

            iteration.outputs.reasks = reasks

        except Exception as e:
            error_message = str(e)
            iteration.outputs.error = error_message
            iteration.outputs.exception = e
            raise e
        return iteration

    # TODO: Refactor this to use inheritance and overrides
    @async_trace(name="call")
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
    async def async_prepare(
        self,
        call_log: Call,
        attempt_number: int,
        *,
        messages: Optional[List[Dict]],
        prompt_params: Optional[Dict] = None,
        api: Optional[Union[PromptCallableBase, AsyncPromptCallableBase]],
    ) -> Awaitable[
        Tuple[ Optional[List[Dict]]]
    ]:
        """Prepare by running pre-processing and input validation.

        Returns:
            messages
        """
        prompt_params = prompt_params or {}
        if api is None:
            raise UserFacingException(ValueError("API must be provided."))

        has_messages_validation = "msg_history" in self.validation_map
        if messages:
            # Runner.prepare_msg_history
            formatted_messages = []

            # Format any variables in the message history with the prompt params.
            for msg in messages:
                msg_copy = copy.deepcopy(msg)
                msg_copy["content"] = msg_copy["content"].format(**prompt_params)
                formatted_messages.append(msg_copy)

            if "messages" in self.validation_map:
                # Runner.validate_msg_history
                messages_str = messages_string(messages)
                inputs = Inputs(
                    llm_output=messages_str,
                )
                iteration = Iteration(inputs=inputs)
                call_log.iterations.insert(0, iteration)
                value, _metadata = await validator_service.async_validate(
                    value=messages_str,
                    metadata=self.metadata,
                    validator_map=self.validation_map,
                    iteration=iteration,
                    disable_tracer=self._disable_tracer,
                    path="messsages",
                )
                validated_messages = validator_service.post_process_validation(
                    value, attempt_number, iteration, OutputTypes.STRING
                )
                validated_messages = cast(str, validated_messages)

                iteration.outputs.validation_response = validated_messages
                if isinstance(validated_messages, ReAsk):
                    raise ValidationError(
                        f"Messages validation failed: "
                        f"{validated_messages}"
                    )
                if validated_messages != messages_str:
                    raise ValidationError("Messages validation failed")
        else:
            raise UserFacingException(
                ValueError("'prompt' or 'msg_history' must be provided.")
            )

        return messages
