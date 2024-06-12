from typing import (
    Any,
    AsyncIterable,
    Dict,
    List,
    Optional,
    Union,
    cast,
)


from guardrails.actions.reask import SkeletonReAsk
from guardrails.classes import ValidationOutcome
from guardrails.classes.history import Call, Inputs, Iteration, Outputs
from guardrails.classes.output_type import OutputTypes
from guardrails.constants import pass_status
from guardrails.llm_providers import (
    AsyncLiteLLMCallable,
    AsyncPromptCallableBase,
    LiteLLMCallable,
    PromptCallableBase,
)
from guardrails.logger import set_scope
from guardrails.messages.messages import Messages
from guardrails.run import StreamRunner
from guardrails.run.async_runner import AsyncRunner
from guardrails.utils.openai_utils import OPENAI_VERSION


class AsyncStreamRunner(AsyncRunner, StreamRunner):
    async def async_run(
        self, call_log: Call, prompt_params: Optional[Dict] = None
    ) -> AsyncIterable[ValidationOutcome]:
        prompt_params = prompt_params or {}

        (
            messages,
            output_schema,
        ) = (
            self.messages,
            self.output_schema,
        )

        result = self.async_step(
            0,
            output_schema,
            call_log,
            api=self.api,
            messages=messages,
            prompt_params=prompt_params,
            output=self.output,
        )
        # FIXME: Where can this be moved to be less verbose? This is an await call on
        # the async generator.
        async for call in result:
            yield call

    # @async_trace(name="step")
    async def async_step(
        self,
        index: int,
        output_schema: Dict[str, Any],
        call_log: Call,
        *,
        api: Optional[AsyncPromptCallableBase],
        messages: Optional[Messages] = None,
        prompt_params: Optional[Dict] = None,
        output: Optional[str] = None,
    ) -> AsyncIterable[ValidationOutcome]:
        prompt_params = prompt_params or {}
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
        set_scope(str(id(iteration)))
        call_log.iterations.push(iteration)
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

        if self.output_type == OutputTypes.STRING:
            async for chunk in stream_output:
                chunk_text = self.get_chunk_text(chunk, api)
                _ = self.is_last_chunk(chunk, api)
                fragment += chunk_text

                parsed_chunk, move_to_next = self.parse(
                    chunk_text, output_schema, verified=verified
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

                reasks, valid_op = self.introspect(
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
                    fragment, output_schema, verified=verified
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

                reasks, valid_op = self.introspect(validated_fragment)
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
        iteration.outputs.validation_response = (
            cast(str, validated_fragment) if validated_fragment else None
        )
        iteration.outputs.guarded_output = valid_op

    # async def async_validate(
    #     self,
    #     iteration: Iteration,
    #     index: int,
    #     parsed_output: Any,
    #     output_schema: Schema,
    #     validate_subschema: bool = False,
    #     stream: Optional[bool] = False,
    # ) -> Optional[Union[Awaitable[ValidationResult], ValidationResult]]:
    #     # FIXME: Subschema is currently broken, it always returns a string from async
    #     # streaming.
    #     # Should return None/empty if fail result?
    #     _ = await output_schema.async_validate(
    #         iteration, parsed_output, self.metadata, attempt_number=index, stream=stream  # noqa
    #     )
    #     try:
    #         return iteration.outputs.validator_logs[-1].validation_result
    #     except IndexError:
    #         return None

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
