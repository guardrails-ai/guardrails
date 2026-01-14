import copy
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast


from guardrails import validator_service
from guardrails.actions.reask import get_reask_setup
from guardrails.classes.execution.guard_execution_options import GuardExecutionOptions
from guardrails.classes.history import Call, Inputs, Iteration, Outputs
from guardrails.classes.output_type import OutputTypes
from guardrails.constants import fail_status
from guardrails.errors import ValidationError
from guardrails.llm_providers import (
    AsyncPromptCallableBase,
    PromptCallableBase,
)
from guardrails.logger import set_scope
from guardrails.prompt import Prompt
from guardrails.prompt.messages import Messages
from guardrails.run.utils import messages_source
from guardrails.schema.rail_schema import json_schema_to_rail_output
from guardrails.schema.validator import schema_validation
from guardrails.hub_telemetry.hub_tracing import trace
from guardrails.types import ModelOrListOfModels, ValidatorMap, MessageHistory
from guardrails.utils.exception_utils import UserFacingException
from guardrails.utils.hub_telemetry_utils import HubTelemetry
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.utils.parsing_utils import (
    coerce_types,
    parse_llm_output,
    prune_extra_keys,
)
from guardrails.utils.prompt_utils import (
    prompt_content_for_schema,
)
from guardrails.actions.reask import NonParseableReAsk, ReAsk, introspect
from guardrails.telemetry import trace_call, trace_step


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

    # Validation Inputs
    output_schema: Dict[str, Any]
    output_type: OutputTypes
    validation_map: ValidatorMap = {}
    metadata: Dict[str, Any]

    # LLM Inputs
    messages: Optional[List[Dict[str, Union[Prompt, str]]]] = None
    base_model: Optional[ModelOrListOfModels]
    exec_options: Optional[GuardExecutionOptions]

    # LLM Calling Details
    api: Optional[PromptCallableBase] = None
    output: Optional[str] = None
    num_reasks: int
    full_schema_reask: bool = False

    # Internal Metrics Collection
    disable_tracer: Optional[bool] = True

    # QUESTION: Are any of these init args actually necessary for initialization?
    # ANSWER: _Maybe_ messages for Prompt initialization
    #   but even that can happen at execution time.
    # TODO: In versions >=0.6.x, remove this class and just execute a Guard functionally
    def __init__(
        self,
        output_type: OutputTypes,
        output_schema: Dict[str, Any],
        num_reasks: int,
        validation_map: ValidatorMap,
        *,
        messages: Optional[List[Dict]] = None,
        api: Optional[PromptCallableBase] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        base_model: Optional[ModelOrListOfModels] = None,
        full_schema_reask: bool = False,
        disable_tracer: Optional[bool] = True,
        exec_options: Optional[GuardExecutionOptions] = None,
    ):
        # Validation Inputs
        self.output_type = output_type
        self.output_schema = output_schema
        self.validation_map = validation_map
        self.metadata = metadata or {}
        self.exec_options = copy.deepcopy(exec_options) or GuardExecutionOptions()

        # LLM Inputs

        stringified_output_schema = prompt_content_for_schema(
            output_type, output_schema, validation_map
        )
        xml_output_schema = json_schema_to_rail_output(
            json_schema=output_schema, validator_map=validation_map
        )

        if messages:
            self.exec_options.messages = messages
            messages_copy = []
            for msg in messages:
                msg_copy = copy.deepcopy(msg)
                msg_copy["content"] = Prompt(
                    msg_copy["content"],
                    output_schema=stringified_output_schema,
                    xml_output_schema=xml_output_schema,
                )
                messages_copy.append(msg_copy)
            self.messages = messages_copy

        self.base_model = base_model

        # LLM Calling Details
        self.api = api
        self.output = output
        self.num_reasks = num_reasks
        self.full_schema_reask = full_schema_reask

        # Internal Metrics Collection
        # Get metrics opt-out from credentials
        self._disable_tracer = disable_tracer

        # Get the HubTelemetry singleton
        self._hub_telemetry = HubTelemetry()
        self._hub_telemetry._enabled = not self._disable_tracer

    @trace(name="/reasks", origin="Runner.__call__")
    def __call__(self, call_log: Call, prompt_params: Optional[Dict] = None) -> Call:
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
            # NOTE: At first glance this seems gratuitous,
            #   but these local variables are reassigned after
            #   calling self.prepare_to_loop
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
                iteration = self.step(
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
                (output_schema, messages) = self.prepare_to_loop(
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

    @trace(name="/step", origin="Runner.step")
    @trace_step
    def step(
        self,
        index: int,
        output_schema: Dict[str, Any],
        call_log: Call,
        *,
        api: Optional[PromptCallableBase],
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
                messages = self.prepare(
                    call_log,
                    messages=messages,
                    prompt_params=prompt_params,
                    api=api,
                    attempt_number=index,
                )

            iteration.inputs.messages = messages

            # Call: run the API.
            llm_response = self.call(messages, api, output)

            iteration.outputs.llm_response_info = llm_response
            raw_output = llm_response.output

            # Parse: parse the output.
            parsed_output, parsing_error = self.parse(raw_output, output_schema)
            if parsing_error or isinstance(parsed_output, ReAsk):
                iteration.outputs.exception = parsing_error  # type: ignore
                iteration.outputs.error = str(parsing_error)
                iteration.outputs.reasks.append(parsed_output)  # type: ignore
            else:
                iteration.outputs.parsed_output = parsed_output

            # Validate: run output validation.
            if parsing_error and isinstance(parsed_output, NonParseableReAsk):
                reasks, _ = self.introspect(parsed_output)
            else:
                # Validate: run output validation.
                validated_output = self.validate(
                    iteration, index, parsed_output, output_schema
                )
                iteration.outputs.validation_response = validated_output

                # Introspect: inspect validated output for reasks.
                reasks, valid_output = self.introspect(validated_output)
                iteration.outputs.guarded_output = valid_output

            iteration.outputs.reasks = list(reasks)

        except Exception as e:
            error_message = str(e)
            iteration.outputs.error = error_message
            iteration.outputs.exception = e
            raise e
        return iteration

    @trace(name="/input_validation", origin="Runner.validate_messages")
    def validate_messages(
        self, call_log: Call, messages: MessageHistory, attempt_number: int
    ) -> None:
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
            value, _metadata = validator_service.validate(
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

    def prepare_messages(
        self,
        call_log: Call,
        messages: MessageHistory,
        prompt_params: Dict,
        attempt_number: int,
    ) -> MessageHistory:
        formatted_messages: MessageHistory = []
        # Format any variables in the message history with the prompt params.
        for msg in messages:
            msg_copy = copy.deepcopy(msg)
            if attempt_number == 0:
                msg_copy["content"] = msg_copy["content"].format(**prompt_params)
            formatted_messages.append(msg_copy)

        # validate messages
        if "messages" in self.validation_map:
            self.validate_messages(call_log, formatted_messages, attempt_number)

        return formatted_messages

    @trace(name="/input_validation", origin="Runner.validate_prompt")
    def validate_prompt(self, call_log: Call, prompt: Prompt, attempt_number: int):
        inputs = Inputs(
            llm_output=prompt.source,
        )
        iteration = Iteration(call_id=call_log.id, index=attempt_number, inputs=inputs)
        call_log.iterations.insert(0, iteration)
        value, _metadata = validator_service.validate(
            value=prompt.source,
            metadata=self.metadata,
            validator_map=self.validation_map,
            iteration=iteration,
            disable_tracer=self._disable_tracer,
            path="prompt",
        )

        validated_prompt = validator_service.post_process_validation(
            value, attempt_number, iteration, OutputTypes.STRING
        )

        iteration.outputs.validation_response = validated_prompt

        if isinstance(validated_prompt, ReAsk):
            raise ValidationError(f"Prompt validation failed: {validated_prompt}")
        elif not validated_prompt or iteration.status == fail_status:
            raise ValidationError("Prompt validation failed")
        return Prompt(cast(str, validated_prompt))

    @trace(name="/input_prep", origin="Runner.prepare")
    def prepare(
        self,
        call_log: Call,
        attempt_number: int,
        *,
        messages: Optional[MessageHistory],
        prompt_params: Optional[Dict] = None,
        api: Optional[Union[PromptCallableBase, AsyncPromptCallableBase]],
    ) -> Optional[MessageHistory]:
        """Prepare by running pre-processing and input validation.

        Returns:
            The message history.
        """
        prompt_params = prompt_params or {}
        if api is None:
            raise UserFacingException(ValueError("API must be provided."))

        if messages:
            messages = self.prepare_messages(
                call_log, messages, prompt_params, attempt_number
            )

        return messages

    @trace(name="/llm_call", origin="Runner.call")
    @trace_call
    def call(
        self,
        messages: Optional[MessageHistory],
        api: Optional[PromptCallableBase],
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
            llm_response = LLMResponse(output=output)
        elif api_fn is None:
            raise ValueError("API or output must be provided.")
        elif messages:
            llm_response = api_fn(messages=messages_source(messages))
        else:
            llm_response = api_fn()

        return llm_response

    def parse(self, output: str, output_schema: Dict[str, Any], **kwargs):
        parsed_output, error = parse_llm_output(output, self.output_type, **kwargs)
        if parsed_output and not error and not isinstance(parsed_output, ReAsk):
            parsed_output = prune_extra_keys(parsed_output, output_schema)
            parsed_output = coerce_types(parsed_output, output_schema)
        return parsed_output, error

    @trace(name="/validation", origin="Runner.validate")
    def validate(
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

        validated_output, metadata = validator_service.validate(
            value=parsed_output,
            metadata=self.metadata,
            validator_map=self.validation_map,
            iteration=iteration,
            disable_tracer=self._disable_tracer,
            path="$",
            stream=stream,
            **kwargs,
        )
        self.metadata.update(metadata)
        validated_output = validator_service.post_process_validation(
            validated_output, attempt_number, iteration, self.output_type
        )

        return validated_output

    def introspect(
        self,
        validated_output: Any,
    ) -> Tuple[Sequence[ReAsk], Optional[Union[str, Dict, List]]]:
        """Introspect the validated output."""
        if validated_output is None:
            return [], None
        reasks, valid_output = introspect(validated_output)

        return reasks, valid_output

    def do_loop(self, attempt_number: int, reasks: Sequence[ReAsk]) -> bool:
        """Determine if we should loop again."""
        if reasks and attempt_number < self.num_reasks:
            return True
        return False

    def prepare_to_loop(
        self,
        reasks: Sequence[ReAsk],
        output_schema: Dict[str, Any],
        *,
        parsed_output: Optional[Union[str, List, Dict, ReAsk]] = None,
        validated_output: Optional[Union[str, List, Dict, ReAsk]] = None,
        prompt_params: Optional[Dict] = None,
    ) -> Tuple[
        Dict[str, Any],
        Optional[Union[List[Dict], Messages]],
    ]:
        """Prepare to loop again."""
        prompt_params = prompt_params or {}
        output_schema, messages = get_reask_setup(
            output_type=self.output_type,
            output_schema=output_schema,
            validation_map=self.validation_map,
            reasks=reasks,
            parsing_response=parsed_output,
            validation_response=validated_output,
            use_full_schema=self.full_schema_reask,
            prompt_params=prompt_params,
            exec_options=self.exec_options,
        )

        return output_schema, messages
