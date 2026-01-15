import json
import os
import inspect
import importlib
from typing import Any, Dict, Optional
from fastapi import HTTPException, Request, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from urllib.parse import unquote_plus
from guardrails import AsyncGuard, Guard
from guardrails.classes import ValidationOutcome
from opentelemetry.trace import Span
from guardrails_api_client import Guard as GuardStruct
from guardrails_api.clients.cache_client import CacheClient
from guardrails_api.clients.memory_guard_client import MemoryGuardClient
from guardrails_api.clients.pg_guard_client import PGGuardClient
from guardrails_api.clients.postgres_client import postgres_is_enabled
from guardrails_api.utils.get_llm_callable import get_llm_callable
from guardrails_api.utils.openai import (
    outcome_to_chat_completion,
    outcome_to_stream_response,
)
from guardrails_api.utils.handle_error import handle_error
from string import Template

# if no pg_host is set, use in memory guards
if postgres_is_enabled():
    guard_client = PGGuardClient()
else:
    guard_client = MemoryGuardClient()
    # Will be defined at runtime
    config = None  # noqa: N806
    for module_name in ("config", "guardrails_api.config"):
        try:
            config = importlib.import_module(module_name)
            break
        except ModuleNotFoundError:
            continue

    if config is not None:
        exports = config.__dir__()
        for export_name in exports:
            export = getattr(config, export_name)
            is_guard = isinstance(export, Guard)
            if is_guard:
                guard_client.create_guard(export)

cache_client = CacheClient()

cache_client.initialize()

router = APIRouter()


def guard_history_is_enabled():
    return os.environ.get("GUARD_HISTORY_ENABLED", "true").lower() == "true"


@router.get("/guards")
@handle_error
async def get_guards():
    guards = guard_client.get_guards()
    return [g.to_dict() for g in guards]


@router.post("/guards")
@handle_error
async def create_guard(request: Request):
    if not postgres_is_enabled():
        raise HTTPException(
            status_code=501,
            detail="Not Implemented POST /guards is not implemented for in-memory guards.",
        )
    # Use from_dict() to handle ValidationType deserialization from JSON strings
    payload = await request.json()
    guard = GuardStruct.from_dict(payload)
    new_guard = guard_client.create_guard(guard)
    return new_guard.to_dict()


@router.get("/guards/{guard_name}")
@handle_error
async def get_guard(guard_name: str, asOf: Optional[str] = None):
    decoded_guard_name = unquote_plus(guard_name)
    guard = guard_client.get_guard(decoded_guard_name, asOf)
    if guard is None:
        raise HTTPException(
            status_code=404,
            detail=f"A Guard with the name {decoded_guard_name} does not exist!",
        )
    return guard.to_dict()


@router.put("/guards/{guard_name}")
@handle_error
async def update_guard(guard_name: str, request: Request):
    if not postgres_is_enabled():
        raise HTTPException(
            status_code=501,
            detail="PUT /<guard_name> is not implemented for in-memory guards.",
        )
    # Use from_dict() to handle ValidationType deserialization from JSON strings
    payload = await request.json()
    guard = GuardStruct.from_dict(payload)
    decoded_guard_name = unquote_plus(guard_name)
    updated_guard = guard_client.upsert_guard(decoded_guard_name, guard)
    return updated_guard.to_dict()


@router.delete("/guards/{guard_name}")
@handle_error
async def delete_guard(guard_name: str):
    if not postgres_is_enabled():
        raise HTTPException(
            status_code=501,
            detail="DELETE /<guard_name> is not implemented for in-memory guards.",
        )
    decoded_guard_name = unquote_plus(guard_name)
    guard = guard_client.delete_guard(decoded_guard_name)
    return guard.to_dict()


@router.post("/guards/{guard_name}/openai/v1/chat/completions")
@handle_error
async def openai_v1_chat_completions(guard_name: str, request: Request):
    payload = await request.json()
    decoded_guard_name = unquote_plus(guard_name)
    guard_struct = guard_client.get_guard(decoded_guard_name)
    if guard_struct is None:
        raise HTTPException(
            status_code=404,
            detail=f"A Guard with the name {decoded_guard_name} does not exist!",
        )

    guard = (
        AsyncGuard.from_dict(guard_struct.to_dict())
        if not isinstance(guard_struct, Guard)
        else guard_struct
    )
    stream = payload.get("stream", False)
    has_tool_gd_tool_call = any(
        tool.get("function", {}).get("name") == "gd_response_tool"
        for tool in payload.get("tools", [])
    )

    if not stream:
        execution = guard(num_reasks=0, **payload)
        if inspect.iscoroutine(execution):
            validation_outcome: ValidationOutcome = await execution
        else:
            validation_outcome: ValidationOutcome = execution

        llm_response = guard.history.last.iterations.last.outputs.llm_response_info
        result = outcome_to_chat_completion(
            validation_outcome=validation_outcome,
            llm_response=llm_response,
            has_tool_gd_tool_call=has_tool_gd_tool_call,
        )
        return JSONResponse(content=result)
    else:

        async def openai_streamer():
            try:
                guard_stream = await guard(num_reasks=0, **payload)
                async for result in guard_stream:
                    chunk = json.dumps(outcome_to_stream_response(validation_outcome=result))
                    yield f"data: {chunk}\n\n"
                yield "\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
                yield "\n"

        return StreamingResponse(openai_streamer(), media_type="text/event-stream")


@router.post("/guards/{guard_name}/validate")
@handle_error
async def validate(guard_name: str, request: Request):
    payload = await request.json()
    openai_api_key = request.headers.get("x-openai-api-key", os.environ.get("OPENAI_API_KEY"))
    decoded_guard_name = unquote_plus(guard_name)
    guard_struct = guard_client.get_guard(decoded_guard_name)

    llm_output = payload.pop("llmOutput", None)
    num_reasks = payload.pop("numReasks", None)
    prompt_params = payload.pop("promptParams", {})
    llm_api = payload.pop("llmApi", None)
    args = payload.pop("args", [])
    stream = payload.pop("stream", False)

    payload["api_key"] = payload.get("api_key", openai_api_key)

    if llm_api is not None:
        llm_api = get_llm_callable(llm_api)
        if openai_api_key is None:
            raise HTTPException(
                status_code=400,
                detail="Cannot perform calls to OpenAI without an api key.",
            )

    guard = guard_struct
    is_async = inspect.iscoroutinefunction(llm_api)

    if not isinstance(guard_struct, Guard):
        if is_async:
            guard = AsyncGuard.from_dict(guard_struct.to_dict())
        else:
            guard: Guard = Guard.from_dict(guard_struct.to_dict())
    elif is_async:
        guard: Guard = AsyncGuard.from_dict(guard_struct.to_dict())

    if llm_api is None and num_reasks and num_reasks > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Cannot perform re-asks without an LLM API. Specify llm_api when calling "
                "guard(...)."
            ),
        )

    if llm_output is not None:
        if stream:
            raise HTTPException(
                status_code=400, detail="Streaming is not supported for parse calls!"
            )
        execution = guard.parse(
            llm_output=llm_output,
            num_reasks=num_reasks,
            prompt_params=prompt_params,
            llm_api=llm_api,
            **payload,
        )
        if inspect.iscoroutine(execution):
            result: ValidationOutcome = await execution
        else:
            result: ValidationOutcome = execution
    else:
        if stream:

            async def guard_streamer():
                call = guard(
                    llm_api=llm_api,
                    prompt_params=prompt_params,
                    num_reasks=num_reasks,
                    stream=stream,
                    *args,
                    **payload,
                )
                is_async = inspect.iscoroutine(call)
                if is_async:
                    guard_stream = await call
                    async for result in guard_stream:
                        validation_output = ValidationOutcome.from_guard_history(guard.history.last)
                        yield validation_output, result
                else:
                    guard_stream = call
                    for result in guard_stream:
                        validation_output = ValidationOutcome.from_guard_history(guard.history.last)
                        yield validation_output, result

            async def validate_streamer(guard_iter):
                try:
                    async for validation_output, result in guard_iter:
                        fragment_dict = result.to_dict()
                        fragment_dict["error_spans"] = [
                            json.dumps({"start": x.start, "end": x.end, "reason": x.reason})
                            for x in guard.error_spans_in_output()
                        ]
                        yield json.dumps(fragment_dict) + "\n"

                    call = guard.history.last
                    final_validation_output = ValidationOutcome(
                        callId=call.id,
                        validation_passed=result.validation_passed,
                        validated_output=result.validated_output,
                        history=guard.history,
                        raw_llm_output=result.raw_llm_output,
                    )
                    final_output_dict = final_validation_output.to_dict()
                    final_output_dict["error_spans"] = [
                        json.dumps({"start": x.start, "end": x.end, "reason": x.reason})
                        for x in guard.error_spans_in_output()
                    ]
                    yield json.dumps(final_output_dict) + "\n"
                except Exception as e:
                    yield json.dumps({"error": {"message": str(e)}}) + "\n"

                if guard_history_is_enabled():
                    serialized_history = [call.to_dict() for call in guard.history]
                    cache_key = f"{guard.name}-{final_validation_output.call_id}"
                    await cache_client.set(cache_key, serialized_history, 300)

            return StreamingResponse(
                validate_streamer(guard_streamer()), media_type="application/json"
            )
        else:
            execution = guard(
                llm_api=llm_api,
                prompt_params=prompt_params,
                num_reasks=num_reasks,
                *args,
                **payload,
            )
            if inspect.iscoroutine(execution):
                result: ValidationOutcome = await execution
            else:
                result: ValidationOutcome = execution
    if guard_history_is_enabled():
        serialized_history = [call.to_dict() for call in guard.history]
        cache_key = f"{guard.name}-{result.call_id}"
        await cache_client.set(cache_key, serialized_history, 300)
    return result.to_dict()


@router.get("/guards/{guard_name}/history/{call_id}")
@handle_error
async def guard_history(guard_name: str, call_id: str):
    cache_key = f"{guard_name}-{call_id}"
    return await cache_client.get(cache_key)


def collect_telemetry(
    *,
    guard: Guard,
    validate_span: Span,
    validation_output: ValidationOutcome,
    prompt_params: Dict[str, Any],
    result: ValidationOutcome,
):
    # Below is all telemetry collection and
    # should have no impact on what is returned to the user
    prompt = guard.history.last.inputs.prompt
    if prompt:
        prompt = Template(prompt).safe_substitute(**prompt_params)
        validate_span.set_attribute("prompt", prompt)

    instructions = guard.history.last.inputs.instructions
    if instructions:
        instructions = Template(instructions).safe_substitute(**prompt_params)
        validate_span.set_attribute("instructions", instructions)

    validate_span.set_attribute("validation_status", guard.history.last.status)
    validate_span.set_attribute("raw_llm_ouput", result.raw_llm_output)

    # Use the serialization from the class instead of re-writing it
    valid_output: str = (
        json.dumps(validation_output.validated_output)
        if isinstance(validation_output.validated_output, dict)
        else str(validation_output.validated_output)
    )
    validate_span.set_attribute("validated_output", valid_output)

    validate_span.set_attribute("tokens_consumed", guard.history.last.tokens_consumed)

    num_of_reasks = (
        guard.history.last.iterations.length - 1 if guard.history.last.iterations.length > 0 else 0
    )
    validate_span.set_attribute("num_of_reasks", num_of_reasks)
