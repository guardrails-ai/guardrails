from contextvars import ContextVar, copy_context
import sys
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    cast,
)

from guardrails.validator_service import AsyncValidatorService
from guardrails.actions.reask import SkeletonReAsk
from guardrails.classes import ValidationOutcome
from guardrails.classes.history import Call, Inputs, Iteration, Outputs
from guardrails.classes.output_type import OutputTypes
from guardrails.llm_providers import (
    AsyncPromptCallableBase,
)
from guardrails.logger import set_scope
from guardrails.run import StreamRunner
from guardrails.run.async_runner import AsyncRunner
from guardrails.telemetry import trace_async_stream_step
from guardrails.hub_telemetry.hub_tracing import async_trace_stream
from guardrails.types import OnFailAction
from guardrails.classes.validation.validation_result import (
    PassResult,
    FailResult,
)


if sys.version_info.minor < 10:
    from guardrails.utils.polyfills import anext


class AsyncStreamRunner(AsyncRunner, StreamRunner):
    # @async_trace_stream(name="/reasks", origin="AsyncStreamRunner.async_run")
    async def async_run(
        self, call_log: Call, prompt_params: Optional[Dict] = None
    ) -> AsyncIterator[ValidationOutcome]:
        prompt_params = prompt_params or {}

        (
            messages,
            output_schema,
        ) = (
            self.messages,
            self.output_schema,
        )

        result = await self.async_step(
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

    @async_trace_stream(name="/step", origin="AsyncStreamRunner.async_step")
    @trace_async_stream_step
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
    ) -> AsyncIterator[ValidationOutcome]:
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
        iteration = Iteration(
            call_id=call_log.id, index=index, inputs=inputs, outputs=outputs
        )
        set_scope(str(id(iteration)))
        call_log.iterations.push(iteration)
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

        llm_response = await self.async_call(messages, api, output)
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
        validation_passed = True

        context = copy_context()
        stream_context_vars: ContextVar[Dict[str, ContextVar[List[str]]]] = ContextVar(
            "stream_context"
        )
        context_vars: Dict[str, ContextVar[List[str]]] = {}
        for k, v in self.validation_map.items():
            if isinstance(v, list):
                for validator in v:
                    property_validation_chunks = ContextVar(
                        f"{k}_{validator.rail_alias}_chunks"
                    )
                    context.run(property_validation_chunks.set, [])
                    context_vars[f"{k}_{validator.rail_alias}"] = (
                        property_validation_chunks  # noqa: E501
                    )
        context.run(stream_context_vars.set, context_vars)

        if self.output_type == OutputTypes.STRING:
            validator_service = AsyncValidatorService(self.disable_tracer)

            next_exists = True
            while next_exists:
                try:
                    chunk = await anext(stream_output)
                    chunk_text = self.get_chunk_text(chunk, api)
                    _ = self.is_last_chunk(chunk, api)

                    fragment += chunk_text

                    results = await validator_service.async_partial_validate(
                        chunk_text,
                        self.metadata,
                        self.validation_map,
                        iteration,
                        "$",
                        "$",
                        True,
                        context=context,
                        context_vars=stream_context_vars,
                    )
                    validators = self.validation_map.get("$", [])

                    # collect the result validated_chunk into validation progress
                    # per validator
                    for result in results:
                        validator_log = result.validator_logs  # type: ignore
                        validator = next(
                            filter(
                                lambda x: x.rail_alias == validator_log.registered_name,
                                validators,
                            ),
                            None,
                        )
                        if (
                            validator_log.validation_result
                            and validator_log.validation_result.validated_chunk
                        ):
                            is_filter = (
                                validator.on_fail_descriptor is OnFailAction.FILTER  # type: ignore
                            )
                            is_refrain = (
                                validator.on_fail_descriptor is OnFailAction.REFRAIN  # type: ignore
                            )
                            if validator_log.validation_result.outcome == "fail":
                                validation_passed = False
                            reasks, valid_op = self.introspect(
                                validator_log.validation_result
                            )
                            if reasks:
                                raise ValueError(
                                    "Reasks are not yet supported with streaming. "
                                    "Please remove reasks from schema or disable"
                                    " streaming."
                                )

                            if isinstance(validator_log.validation_result, PassResult):
                                chunk = validator_log.validation_result.validated_chunk
                            elif isinstance(
                                validator_log.validation_result, FailResult
                            ):
                                if is_filter or is_refrain:
                                    refrain_triggered = True
                                    chunk = ""
                                else:
                                    chunk = validator_service.perform_correction(
                                        validator_log.validation_result,
                                        validator_log.validation_result.validated_chunk,
                                        validator,  # type: ignore
                                        rechecked_value=None,
                                    )  # type: ignore

                            if validator_log.validator_name not in validation_progress:
                                validation_progress[validator_log.validator_name] = ""

                            validation_progress[validator_log.validator_name] += chunk
                    # if there is an entry for every validator
                    # run a merge and emit a validation outcome
                    if (
                        len(validation_progress) == len(validators)
                        or len(validators) == 0
                    ):
                        if refrain_triggered:
                            current = ""
                        else:
                            merge_chunks = []
                            for piece in validation_progress:
                                merge_chunks.append(validation_progress[piece])

                            current = validator_service.multi_merge(
                                fragment, merge_chunks
                            )

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

                except StopIteration:
                    next_exists = False
                except StopAsyncIteration:
                    next_exists = False
                except Exception as e:
                    raise e
                finally:
                    # reset all context vars
                    for context_var in context_vars.values():
                        token = context.run(context_var.set, [])
                        context.run(context_var.reset, token)
                    token = context.run(stream_context_vars.set, {})
                    context.run(stream_context_vars.reset, token)

            # if theres anything left merge and emit a chunk
            if len(validation_progress) > 0:
                merge_chunks = []
                for piece in validation_progress:
                    merge_chunks.append(validation_progress[piece])

                current = validator_service.multi_merge(fragment, merge_chunks)
                yield ValidationOutcome(
                    call_id=call_log.id,  # type: ignore
                    raw_llm_output=fragment,
                    validated_output=current,
                    validation_passed=validation_passed,
                )
        else:
            next_exists = True
            while next_exists:
                try:
                    chunk = await anext(stream_output)
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
                        context=context,
                        context_vars=stream_context_vars,
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
                        validated_output=validated_fragment,
                        validation_passed=validated_fragment is not None,
                    )
                    fragment = ""
                except StopIteration:
                    next_exists = False
                except StopAsyncIteration:
                    next_exists = False
                except Exception as e:
                    raise e
                finally:
                    # reset all context vars
                    for context_var in context_vars.values():
                        token = context.run(context_var.set, [])
                        context.run(context_var.reset, token)
                    token = context.run(stream_context_vars.set, {})
                    context.run(stream_context_vars.reset, token)

        iteration.outputs.raw_output = fragment
        # FIXME: Handle case where parsing continuously fails/is a reask
        iteration.outputs.parsed_output = parsed_fragment or fragment  # type: ignore
        iteration.outputs.validation_response = validation_response
        iteration.outputs.guarded_output = valid_op
