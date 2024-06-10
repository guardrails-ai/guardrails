import copy
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel

from guardrails.classes.history import Call, Inputs, Iteration, Outputs
from guardrails.datatypes import verify_metadata_requirements
from guardrails.errors import ValidationError
from guardrails.llm_providers import AsyncPromptCallableBase, PromptCallableBase
from guardrails.logger import set_scope
from guardrails.prompt import Instructions, Prompt
from guardrails.run.utils import msg_history_source, msg_history_string
from guardrails.schema import Schema, StringSchema
from guardrails.utils.exception_utils import UserFacingException
from guardrails.utils.hub_telemetry_utils import HubTelemetry
from guardrails.utils.llm_response import LLMResponse
from guardrails.utils.reask_utils import NonParseableReAsk, ReAsk
from guardrails.utils.telemetry_utils import trace


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
        messages: Optional[List[Dict]] = None,
        api: Optional[PromptCallableBase] = None,
        messages_schema: Optional[StringSchema] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        base_model: Optional[
            Union[Type[BaseModel], Type[List[Type[BaseModel]]]]
        ] = None,
        full_schema_reask: bool = False,
        disable_tracer: Optional[bool] = True,
    ):

        if messages:
            messages = copy.deepcopy(messages)
            messages_copy = []
            for msg in messages:
                msg["content"] = Prompt(
                    msg["content"], output_schema=output_schema.transpile()
                )
                messages_copy.append(msg)
            self.messages = messages_copy
        else:
            self.messages = None

        self.api = api
        self.messages_schema = messages_schema
        self.output_schema = output_schema
        self.num_reasks = num_reasks
        self.metadata = metadata or {}
        self.output = output
        self.base_model = base_model
        self.full_schema_reask = full_schema_reask

        # Get metrics opt-out from credentials
        self._disable_tracer = disable_tracer

        if not self._disable_tracer:
            # Get the HubTelemetry singleton
            self._hub_telemetry = HubTelemetry()

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

            (
                messages,
                messages_schema,
                output_schema,
            ) = (
                self.messages,
                self.messages_schema,
                self.output_schema,
            )
            index = 0
            for index in range(self.num_reasks + 1):
                # Run a single step.
                iteration = self.step(
                    index=index,
                    api=self.api,
                    instructions=instructions,
                    prompt=prompt,
                    messages=messages,
                    prompt_params=prompt_params,
                    messages_schema=messages_schema,
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
                    call_log.validation_response,
                    output_schema,
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
            call_log._exception = e.original_exception
            raise e.original_exception
        except Exception as e:
            # Because Pydantic v1 doesn't respect property setters
            call_log._exception = e
            raise e
        return call_log

    @trace(name="step")
    def step(
        self,
        index: int,
        api: Optional[PromptCallableBase],
        messages: Optional[List[Dict]],
        prompt_params: Dict,
        messages_schema: Optional[StringSchema],
        output_schema: Schema,
        call_log: Call,
        output: Optional[str] = None,
    ) -> Iteration:
        """Run a full step."""
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
                instructions, prompt, messages = self.prepare(
                    call_log,
                    index,
                    messages,
                    prompt_params,
                    api,
                    messages_schema,
                    output_schema,
                )

            iteration.inputs.messages = messages

            # Call: run the API.
            llm_response = self.call(
                index, messages, api, output
            )

            iteration.outputs.llm_response_info = llm_response
            raw_output = llm_response.output

            # Parse: parse the output.
            parsed_output, parsing_error = self.parse(index, raw_output, output_schema)
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

    def validate_messages(
        self,
        call_log: Call,
        messages: List[Dict],
        messages_schema: StringSchema,
    ):
        msg_str = msg_history_string(messages)
        inputs = Inputs(
            llm_output=msg_str,
        )
        iteration = Iteration(inputs=inputs)
        call_log.iterations.insert(0, iteration)
        validated_messages = messages_schema.validate(
            iteration, msg_str, self.metadata, disable_tracer=self._disable_tracer
        )
        iteration.outputs.validation_response = validated_messages
        if isinstance(validated_messages, ReAsk):
            raise ValidationError(
                f"Message history validation failed: " f"{validated_messages}"
            )
        if validated_messages != msg_str:
            raise ValidationError("Message history validation failed")

    def prepare_messages(
        self,
        call_log: Call,
        messages: List[Dict],
        prompt_params: Dict,
        messages_schema: Optional[StringSchema],
    ):
        messages = copy.deepcopy(messages)
        # Format any variables in the message history with the prompt params.
        for msg in messages:
            msg["content"] = msg["content"].format(**prompt_params)

        # validate messages
        if messages_schema is not None:
            self.validate_messages(call_log, messages, messages_schema)

        return messages

    def prepare(
        self,
        call_log: Call,
        index: int,
        messages: Optional[List[Dict]],
        prompt_params: Dict,
        api: Optional[Union[PromptCallableBase, AsyncPromptCallableBase]],
        messages_schema: Optional[StringSchema],
        output_schema: Schema, #TODO figure out why this is unused in the messages context
    ) -> Tuple[Optional[Instructions], Optional[Prompt], Optional[List[Dict]]]:
        """Prepare by running pre-processing and input validation.

        Returns:
            The instructions, prompt, and message history.
        """
        if api is None:
            raise UserFacingException(ValueError("API must be provided."))

        if prompt_params is None:
            prompt_params = {}

        if messages:
            messages = self.prepare_messages(
                call_log, messages, prompt_params, messages_schema
            )

        return messages

    @trace(name="call")
    def call(
        self,
        index: int,
        messages: Optional[List[Dict[str, str]]],
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
        elif messages:
            llm_response = api_fn(messages=msg_history_source(messages))
        else:
            llm_response = api_fn()

        return llm_response

    def parse(
        self,
        index: int,
        output: str,
        output_schema: Schema,
    ):
        parsed_output, error = output_schema.parse(output)
        return parsed_output, error

    def validate(
        self,
        iteration: Iteration,
        index: int,
        parsed_output: Any,
        output_schema: Schema,
        stream: Optional[bool] = False,
        **kwargs,
    ):
        """Validate the output."""
        if isinstance(output_schema, StringSchema):
            validated_output = output_schema.validate(
                iteration,
                parsed_output,
                self.metadata,
                index,
                self._disable_tracer,
                stream,
                **kwargs,
            )
        else:
            validated_output = output_schema.validate(
                iteration,
                parsed_output,
                self.metadata,
                attempt_number=index,
                disable_tracer=self._disable_tracer,
                **kwargs,
            )

        return validated_output

    def introspect(
        self,
        index: int,
        validated_output: Any,
        output_schema: Schema,
    ) -> Tuple[Sequence[ReAsk], Optional[Union[str, Dict]]]:
        """Introspect the validated output."""
        if validated_output is None:
            return [], None
        reasks, valid_output = output_schema.introspect(validated_output)

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
    ) -> Tuple[Prompt, Optional[Instructions], Schema, Optional[List[Dict]]]:
        """Prepare to loop again."""
        output_schema = output_schema.get_reask_setup(
            reasks=reasks,
            original_response=validated_output,
            use_full_schema=self.full_schema_reask,
            prompt_params=prompt_params,
        )

        messages = None  # clear msg history for reasking
        return output_schema, messages
