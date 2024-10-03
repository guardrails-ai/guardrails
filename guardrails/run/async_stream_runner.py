from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Union,
    cast,
)

from guardrails.merge import merge
from guardrails.validator_service import AsyncValidatorService
from guardrails.actions.reask import SkeletonReAsk
from guardrails.classes import ValidationOutcome
from guardrails.classes.history import Call, Inputs, Iteration, Outputs
from guardrails.classes.output_type import OutputTypes
from guardrails.llm_providers import (
    AsyncLiteLLMCallable,
    AsyncPromptCallableBase,
    LiteLLMCallable,
    OpenAICallable,
    OpenAIChatCallable,
    PromptCallableBase,
)
from guardrails.logger import set_scope
from guardrails.prompt import Instructions, Prompt
from guardrails.run import StreamRunner
from guardrails.run.async_runner import AsyncRunner
from guardrails.telemetry import trace_async_stream_step
from guardrails.hub_telemetry.hub_tracing import async_trace_stream
from guardrails.types import OnFailAction
from guardrails.classes.validation.validation_result import (
    PassResult,
)

class AsyncStreamRunner(AsyncRunner, StreamRunner):
    # @async_trace_stream(name="/reasks", origin="AsyncStreamRunner.async_run")
    async def async_run(
        self, call_log: Call, prompt_params: Optional[Dict] = None
    ) -> AsyncIterator[ValidationOutcome]:
        prompt_params = prompt_params or {}

        (
            instructions,
            prompt,
            msg_history,
            output_schema,
        ) = (
            self.instructions,
            self.prompt,
            self.msg_history,
            self.output_schema,
        )

        result = await self.async_step(
            0,
            output_schema,
            call_log,
            api=self.api,
            instructions=instructions,
            prompt=prompt,
            msg_history=msg_history,
            prompt_params=prompt_params,
            output=self.output,
        )
        # FIXME: Where can this be moved to be less verbose? This is an await call on
        # the async generator.
        async for call in result:
            yield call

    @async_trace_stream(name="/step", origin="AsyncStreamRunner.async_step")
    @trace_async_stream_step
    async def async_step(
        self,
        index: int,
        output_schema: Dict[str, Any],
        call_log: Call,
        *,
        api: Optional[AsyncPromptCallableBase],
        instructions: Optional[Instructions],
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict]] = None,
        prompt_params: Optional[Dict] = None,
        output: Optional[str] = None,
    ) -> AsyncIterator[ValidationOutcome]:
        prompt_params = prompt_params or {}
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
            stream=True,
        )
        outputs = Outputs()
        iteration = Iteration(
            call_id=call_log.id, index=index, inputs=inputs, outputs=outputs
        )
        set_scope(str(id(iteration)))
        call_log.iterations.push(iteration)
        if output is not None:
            instructions = None
            prompt = None
            msg_history = None
        else:
            instructions, prompt, msg_history = await self.async_prepare(
                call_log,
                instructions=instructions,
                prompt=prompt,
                msg_history=msg_history,
                prompt_params=prompt_params,
                api=api,
                attempt_number=index,
            )

        iteration.inputs.prompt = prompt
        iteration.inputs.instructions = instructions
        iteration.inputs.msg_history = msg_history

        llm_response = await self.async_call(
            instructions, prompt, msg_history, api, output
        )
        iteration.outputs.llm_response_info = llm_response
        stream_output = llm_response.async_stream_output
        if not stream_output:
            raise ValueError(
                "No async stream was returned from the API. Please check that "
                "the API is returning an async generator."
            )

        fragment = ""
        parsed_fragment, validated_fragment, valid_op = None, None, None
        verified = set()
        validation_response = ""
        validation_progress = {}
        refrain_triggered = False
        if self.output_type == OutputTypes.STRING:
            async for chunk in stream_output:
                chunk_text = self.get_chunk_text(chunk, api)
                _ = self.is_last_chunk(chunk, api)
                parsed_chunk, move_to_next = self.parse(
                    chunk_text, output_schema, verified=verified
                )
                if move_to_next:
                    continue
                fragment += chunk_text

                validator_service = AsyncValidatorService(self.disable_tracer)
                results = await validator_service.async_partial_validate(
                    chunk_text, self.metadata, self.validation_map, iteration, "$", "$", True,
                )

                # collect the result validated_chunk into validation progress per validator
                for result in results:
                    validator_log = result.validator_logs
                    validators = self.validation_map["$"]
                    validator = next(filter(lambda x: x.rail_alias == validator_log.registered_name, validators), None)

                    if (validator_log.validation_result and validator_log.validation_result.validated_chunk):
                        is_filter = validator.on_fail_descriptor is OnFailAction.FILTER
                        is_refrain = validator.on_fail_descriptor is OnFailAction.REFRAIN

                        reasks, valid_op = self.introspect(validator_log.validation_result)

                        if is_filter or is_refrain:
                            refrain_triggered = True
                            chunk = ""

                        if reasks:
                            raise ValueError(
                                "Reasks are not yet supported with streaming. Please "
                                "remove reasks from schema or disable streaming."
                            )
                        elif isinstance(validator_log.validation_result, PassResult):
                            chunk = validator_log.validation_result.validated_chunk
                        else:
                            chunk = validator_service.perform_correction(
                                validator_log.validation_result,
                                validator_log.validation_result.validated_chunk,
                                validator,
                                rechecked_value=None,
                            )
                        if not hasattr(validation_progress, validator_log.validator_name):
                            validation_progress[validator_log.validator_name] = ""
                        
                        validation_progress[validator_log.validator_name] += chunk

                # if there is an entry for every validator
                # run a merge and emit a validation outcome
                if len(validation_progress) == len(validators):
                    if refrain_triggered:
                        current = ""
                    else:q
                        merge_chunks = []
                        for piece in validation_progress:
                            merge_chunks.append(validation_progress[piece])

                        current = merge_chunks.pop()
                        while len(merge_chunks) > 0:
                            nextval = merge_chunks.pop()
                            current = merge(current, nextval, fragment)

                    vo = ValidationOutcome(
                        call_id=call_log.id,  # type: ignore
                        raw_llm_output=fragment,
                        validated_output=current,
                        validation_passed=True,
                    )
                    fragment = ""
                    validation_progress = {}
                    refrain_triggered = False

                    yield vo

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

                if self.output_type == OutputTypes.LIST:
                    validation_response = cast(list, validated_fragment)
                else:
                    validation_response = cast(dict, validated_fragment)
                yield ValidationOutcome(
                    call_id=call_log.id,  # type: ignore
                    raw_llm_output=fragment,
                    validated_output=chunk_text,
                    validation_passed=validated_fragment is not None,
                )

        iteration.outputs.raw_output = fragment
        # FIXME: Handle case where parsing continuously fails/is a reask
        iteration.outputs.parsed_output = parsed_fragment or fragment  # type: ignore
        iteration.outputs.validation_response = validation_response
        iteration.outputs.guarded_output = valid_op

    def get_chunk_text(self, chunk: Any, api: Union[PromptCallableBase, None]) -> str:
        """Get the text from a chunk."""
        chunk_text = ""
        if isinstance(api, OpenAICallable):
            finished = chunk.choices[0].finish_reason
            content = chunk.choices[0].text
            if not finished and content:
                chunk_text = content
        elif isinstance(api, OpenAIChatCallable):
            finished = chunk.choices[0].finish_reason
            content = chunk.choices[0].delta.content
            if not finished and content:
                chunk_text = content
        elif isinstance(api, LiteLLMCallable):
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
