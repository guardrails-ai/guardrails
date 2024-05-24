import copy
from functools import partial
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel

from guardrails.classes.history import Call, Inputs, Iteration, Outputs
from guardrails.classes.output_type import OT
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.datatypes import verify_metadata_requirements
from guardrails.errors import ValidationError
from guardrails.llm_providers import AsyncPromptCallableBase, PromptCallableBase
from guardrails.prompt import Instructions, Prompt
from guardrails.run.runner import Runner
from guardrails.run.utils import msg_history_source, msg_history_string
from guardrails.schema import Schema, StringSchema
from guardrails.utils.llm_response import LLMResponse
from guardrails.utils.reask_utils import ReAsk, SkeletonReAsk
from guardrails.utils.telemetry_utils import async_trace


class AsyncStreamRunner(Runner):
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
        if prompt_params is None:
            prompt_params = {}

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

        result = self.async_step(
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
        async for call in result:
            yield ValidationOutcome[OT].from_guard_history(call)

    # @async_trace(name="step")
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
    ) -> AsyncGenerator[ValidationOutcome[OT], None]:
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

        iteration.inputs.prompt = prompt
        iteration.inputs.instructions = instructions
        iteration.inputs.msg_history = msg_history

        llm_response = await self.async_call(
            index, instructions, prompt, msg_history, api, output
        )
        try:
            stream = llm_response.completion_stream
        except AttributeError:
            stream = llm_response.stream_output
        if stream is None:
            raise ValueError(
                "No stream was returned from the API. Please check that "
                "the API is returning an async generator."
            )

        fragment = ""
        parsed_fragment, validated_fragment, valid_op = None, None, None
        verified = set()

        if isinstance(output_schema, StringSchema):
            async for chunk in stream:
                chunk_text = self.get_chunk_text(chunk, api)
                finished = self.is_last_chunk(chunk, api)
                fragment += chunk_text

                parsed_chunk, move_to_next = self.parse(
                    index, chunk_text, output_schema, verified
                )
                if move_to_next:
                    continue
                validated_result = await self.async_validate(
                    iteration,
                    index,
                    parsed_chunk,
                    output_schema,
                    True,
                    validate_subschema=True,
                    remainder=finished,
                )
                if isinstance(validated_result, SkeletonReAsk):
                    raise ValueError(
                        "Received fragment schema is an invalid sub-schema "
                        "of the expected output JSON schema."
                    )

                reasks, valid_op = await self.introspect(
                    index, validated_result, output_schema
                )
                if reasks:
                    raise ValueError(
                        "Reasks are not yet supported with streaming. Please "
                        "remove reasks from schema or disable streaming."
                    )

                yield ValidationOutcome(
                    raw_llm_output=chunk_text,
                    validated_output=validated_result,
                    validation_passed=validated_result is not None,
                )
        else:
            async for chunk in stream:
                chunk_text = self.get_chunk_text(chunk, api)
                fragment += chunk_text

                parsed_fragment, move_to_next = self.parse(
                    index, fragment, output_schema, verified
                )
                if move_to_next:
                    continue
                validated_fragment = await self.async_validate(
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

                reasks, valid_op = await self.introspect(
                    index, validated_fragment, output_schema
                )
                if reasks:
                    raise ValueError(
                        "Reasks are not yet supported with streaming. Please "
                        "remove reasks from schema or disable streaming."
                    )

                yield ValidationOutcome(
                    raw_llm_output=fragment,
                    validated_output=validated_fragment,
                    validation_passed=validated_fragment is not None,
                )

        iteration.outputs.raw_output = fragment
        iteration.outputs.parsed_output = parsed_fragment
        iteration.outputs.validation_response = validated_fragment
        iteration.outputs.guarded_output = valid_op

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
        validate_subschema: bool = False,
    ):
        if validate_subschema:
            validated_output = await output_schema.async_validate_subschema(
                iteration, parsed_output, self.metadata, attempt_number=index
            )
        else:
            validated_output = await output_schema.async_validate(
                iteration, parsed_output, self.metadata, attempt_number=index
            )

        return validated_output

    async def introspect(
        self,
        index: int,
        validated_output: Any,
        output_schema: Schema,
    ) -> Tuple[List[ReAsk], Any]:
        # Introspect: inspect validated output for reasks.
        reasks, valid_output = await output_schema.async_introspect(
            validated_output, self.metadata, attempt_number=index + 1
        )
        return reasks, valid_output

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
        if api is None:
            raise ValueError("API must be provided.")

        if prompt_params is None:
            prompt_params = {}

        if msg_history:
            msg_history = copy.deepcopy(msg_history)
            for msg in msg_history:
                msg["content"] = msg["content"].format(**prompt_params)

            prompt, instructions = None, None

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

            if instructions is not None and isinstance(instructions, Instructions):
                instructions = instructions.format(**prompt_params)

            instructions, prompt = output_schema.preprocess_prompt(
                api, instructions, prompt
            )

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
