import copy
from functools import partial
from typing import (
    Any,
    AsyncIterable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    Awaitable,
)

from pydantic import BaseModel

from guardrails.classes import ValidationOutcome
from guardrails.classes.history import Call, Inputs, Iteration, Outputs
from guardrails.constants import pass_status
from guardrails.datatypes import verify_metadata_requirements
from guardrails.errors import ValidationError
from guardrails.llm_providers import (
    AsyncLiteLLMCallable,
    AsyncPromptCallableBase,
    LiteLLMCallable,
    PromptCallableBase,
)
from guardrails.prompt import Instructions, Prompt
from guardrails.run import StreamRunner
from guardrails.run.utils import msg_history_source, msg_history_string
from guardrails.schema import Schema, StringSchema
from guardrails.utils.llm_response import LLMResponse
from guardrails.utils.openai_utils import OPENAI_VERSION
from guardrails.utils.reask_utils import ReAsk, SkeletonReAsk
from guardrails.utils.telemetry_utils import async_trace
from guardrails.validator_base import ValidationResult


class AsyncStreamRunner(StreamRunner):
    def __init__(
        self,
        output_schema: Schema,
        num_reasks: int,
        messages: Optional[List[Dict]] = None,
        api: Optional[AsyncPromptCallableBase] = None,
        messages_schema: Optional[StringSchema] = None,
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
            messages=messages,
            api=api,
            messages_schema=messages_schema,
            metadata=metadata,
            output=output,
            base_model=base_model,
            full_schema_reask=full_schema_reask,
            disable_tracer=disable_tracer,
        )
        self.api: Optional[AsyncPromptCallableBase] = api

    async def async_run(
        self, call_log: Call, prompt_params: Optional[Dict] = None
    ) -> AsyncIterable[ValidationOutcome]:
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
            messages,
            messages_schema,
            output_schema,
        ) = (
            self.messages,
            self.messages_schema,
            self.output_schema,
        )

        result = self.async_step(
            index=0,
            api=self.api,
            messages=messages,
            prompt_params=prompt_params,
            messages_schema=messages_schema,
            output_schema=output_schema,
            output=self.output,
            call_log=call_log,
        )
        # FIXME: Where can this be moved to be less verbose? This is an await call on
        # the async generator.
        async for call in result:
            yield call

    # @async_trace(name="step")
    async def async_step(
        self,
        index: int,
        api: Optional[AsyncPromptCallableBase],
        messages: Optional[List[Dict]],
        prompt_params: Dict,
        messages_schema: Optional[StringSchema],
        output_schema: Schema,
        call_log: Call,
        output: Optional[str] = None,
    ) -> AsyncIterable[ValidationOutcome]:
        inputs = Inputs(
            llm_api=api,
            llm_output=output,
            messages=messages,
            prompt_params=prompt_params,
            num_reasks=self.num_reasks,
            metadata=self.metadata,
            full_schema_reask=self.full_schema_reask,
            stream=True,
        )
        outputs = Outputs()
        iteration = Iteration(inputs=inputs, outputs=outputs)
        call_log.iterations.push(iteration)
        if output:
            messages = None
        else:
            messages = await self.async_prepare(
                call_log,
                index,
                messages,
                prompt_params,
                api,
                messages_schema,
                output_schema,
            )

        iteration.inputs.messages = messages

        llm_response = await self.async_call(
            index, messages, api, output
        )
        stream_output = llm_response.async_stream_output
        if not stream_output:
            raise ValueError(
                "No async stream was returned from the API. Please check that "
                "the API is returning an async generator."
            )

        fragment = ""
        parsed_fragment, validated_fragment, valid_op = None, None, None
        verified = set()

        if isinstance(output_schema, StringSchema):
            async for chunk in stream_output:
                chunk_text = self.get_chunk_text(chunk, api)
                _ = self.is_last_chunk(chunk, api)
                fragment += chunk_text

                parsed_chunk, move_to_next = self.parse(
                    index, chunk_text, output_schema, verified
                )
                if move_to_next:
                    continue
                validated_fragment = await self.async_validate(
                    iteration,
                    index,
                    parsed_chunk,
                    output_schema,
                    validate_subschema=True,
                    stream=True,
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
                passed = call_log.status == pass_status
                yield ValidationOutcome(
                    raw_llm_output=chunk_text,
                    validated_output=validated_fragment,
                    validation_passed=passed,
                )
        else:
            async for chunk in stream_output:
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
                    validated_output=chunk_text,
                    validation_passed=validated_fragment is not None,
                )

        iteration.outputs.raw_output = fragment
        iteration.outputs.parsed_output = parsed_fragment
        iteration.outputs.guarded_output = valid_op
        iteration.outputs.validation_response = (
            cast(str, validated_fragment) if validated_fragment else None
        )

    @async_trace(name="call")
    async def async_call(
        self,
        index: int,
        messages: Optional[List[Dict]],
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
        elif messages:
            llm_response = await api_fn(messages=msg_history_source(messages))
        else:
            llm_response = await api_fn()
        return llm_response

    async def async_validate(
        self,
        iteration: Iteration,
        index: int,
        parsed_output: Any,
        output_schema: Schema,
        validate_subschema: bool = False,
        stream: Optional[bool] = False,
    ) -> Optional[Union[Awaitable[ValidationResult], ValidationResult]]:
        # FIXME: Subschema is currently broken, it always returns a string from async
        # streaming.
        # Should return None/empty if fail result?
        _ = await output_schema.async_validate(
            iteration, parsed_output, self.metadata, attempt_number=index, stream=stream
        )
        try:
            return iteration.outputs.validator_logs[-1].validation_result
        except IndexError:
            return None

    async def introspect(
        self,
        index: int,
        validated_output: Any,
        output_schema: Schema,
    ) -> Tuple[Sequence[ReAsk], Any]:
        # Introspect: inspect validated output for reasks.
        if validated_output is None:
            return [], None
        reasks, valid_output = output_schema.introspect(validated_output)

        return reasks, valid_output

    async def async_prepare(
        self,
        call_log: Call,
        index: int,
        messages: Optional[List[Dict]],
        prompt_params: Dict,
        api: Optional[Union[PromptCallableBase, AsyncPromptCallableBase]],
        messages_schema: Optional[StringSchema],
        output_schema: Schema,
    ) -> Tuple[Optional[List[Dict]]]:
        if api is None:
            raise ValueError("API must be provided.")

        if prompt_params is None:
            prompt_params = {}

        if messages:
            messages = copy.deepcopy(messages)
            for msg in messages:
                msg["content"] = msg["content"].format(**prompt_params)

            prompt, instructions = None, None

            if messages_schema is not None:
                msg_str = msg_history_string(messages)
                inputs = Inputs(
                    llm_output=msg_str,
                )
                iteration = Iteration(inputs=inputs)
                call_log.iterations.insert(0, iteration)
                validated_messages = await messages_schema.async_validate(
                    iteration, msg_str, self.metadata
                )
                if isinstance(validated_messages, ReAsk):
                    raise ValidationError(
                        f"Message validation failed: "
                        f"{validated_messages}"
                    )
                if validated_messages != msg_str:
                    raise ValidationError("Message validation failed")
        else:
            raise ValueError("messages must be provided.")

        return messages

    def get_chunk_text(self, chunk: Any, api: Union[PromptCallableBase, None]) -> str:
        """Get the text from a chunk."""
        chunk_text = ""
        if isinstance(api, LiteLLMCallable):
            finished = chunk.choices[0].finish_reason
            content = chunk.choices[0].delta.content
            if not finished and content:
                chunk_text = content
        elif isinstance(api, AsyncLiteLLMCallable):
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

    def is_last_chunk(self, chunk: Any, api: Union[PromptCallableBase, None]) -> bool:
        """Detect if chunk is final chunk."""
        if isinstance(api, LiteLLMCallable):
            finished = chunk.choices[0].finish_reason
            return finished is not None
        else:
            try:
                finished = chunk.choices[0].finish_reason
                return finished is not None
            except (AttributeError, TypeError):
                return False
