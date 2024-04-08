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
        prompt: Optional[Union[str, Prompt]] = None,
        instructions: Optional[Union[str, Instructions]] = None,
        msg_history: Optional[List[Dict]] = None,
        api: Optional[PromptCallableBase] = None,
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

            # Figure out if we need to include instructions in the prompt.
            include_instructions = not (
                self.instructions is None and self.msg_history is None
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
            index = 0
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
                    call_log.validation_response,
                    output_schema,
                    prompt_params=prompt_params,
                    include_instructions=include_instructions,
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
            iteration, msg_str, self.metadata, disable_tracer=self._disable_tracer
        )
        iteration.outputs.validation_response = validated_msg_history
        if isinstance(validated_msg_history, ReAsk):
            raise ValidationError(
                f"Message history validation failed: " f"{validated_msg_history}"
            )
        if validated_msg_history != msg_str:
            raise ValidationError("Message history validation failed")

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
            iteration,
            prompt.source,
            self.metadata,
            disable_tracer=self._disable_tracer,
        )
        iteration.outputs.validation_response = validated_prompt
        if validated_prompt is None:
            raise ValidationError("Prompt validation failed")
        if isinstance(validated_prompt, ReAsk):
            raise ValidationError(f"Prompt validation failed: {validated_prompt}")
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
            iteration,
            instructions.source,
            self.metadata,
            disable_tracer=self._disable_tracer,
        )
        iteration.outputs.validation_response = validated_instructions
        if validated_instructions is None:
            raise ValidationError("Instructions validation failed")
        if isinstance(validated_instructions, ReAsk):
            raise ValidationError(
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

        return instructions, prompt, msg_history

    @trace(name="call")
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
        elif msg_history:
            llm_response = api_fn(msg_history=msg_history_source(msg_history))
        elif prompt and instructions:
            llm_response = api_fn(prompt.source, instructions=instructions.source)
        elif prompt:
            llm_response = api_fn(prompt.source)
        else:
            raise ValueError("'prompt' or 'msg_history' must be provided.")

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
        **kwargs,
    ):
        """Validate the output."""
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
