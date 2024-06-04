import contextvars
import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

from guardrails import Guard
from guardrails.classes import OT, ValidationOutcome
from guardrails.classes.history import Call
from guardrails.classes.history.call_inputs import CallInputs
from guardrails.llm_providers import get_async_llm_ask, model_is_supported_server_side
from guardrails.logger import set_scope
from guardrails.run import AsyncRunner
from guardrails.stores.context import set_call_kwargs, set_tracer, set_tracer_context


class AsyncGuard(Guard):
    """The Guard class.

    This class one of the main entry point for using Guardrails. It is
    initialized from one of the following class methods:

    - `from_rail`
    - `from_rail_string`
    - `from_pydantic`
    - `from_string`

    The `__call__`
    method functions as a wrapper around LLM APIs. It takes in an Async LLM
    API, and optional prompt parameters, and returns the raw output stream from
    the LLM and the validated output stream.
    """

    async def __call__(
        self,
        llm_api: Union[Callable, Callable[[Any], Awaitable[Any]]],
        prompt_params: Optional[Dict] = None,
        num_reasks: Optional[int] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> Union[
        Union[ValidationOutcome[OT], Iterable[ValidationOutcome[OT]]],
        Awaitable[ValidationOutcome[OT]],
    ]:
        """Call the LLM and validate the output. Pass an async LLM API to
        return a coroutine.

        Args:
            llm_api: The LLM API to call
                     (e.g. openai.Completion.create or openai.Completion.acreate)
            prompt_params: The parameters to pass to the prompt.format() method.
            num_reasks: The max times to re-ask the LLM for invalid output.
            prompt: The prompt to use for the LLM.
            instructions: Instructions for chat models.
            msg_history: The message history to pass to the LLM.
            metadata: Metadata to pass to the validators.
            full_schema_reask: When reasking, whether to regenerate the full schema
                               or just the incorrect values.
                               Defaults to `True` if a base model is provided,
                               `False` otherwise.

        Returns:
            The raw text output from the LLM and the validated output.
        """

        async def __call(
            self,
            llm_api: Union[Callable, Callable[[Any], Awaitable[Any]]],
            prompt_params: Optional[Dict] = None,
            num_reasks: Optional[int] = None,
            prompt: Optional[str] = None,
            instructions: Optional[str] = None,
            msg_history: Optional[List[Dict]] = None,
            metadata: Optional[Dict] = None,
            full_schema_reask: Optional[bool] = None,
            *args,
            **kwargs,
        ):
            if metadata is None:
                metadata = {}
            if full_schema_reask is None:
                full_schema_reask = self.base_model is not None
            if prompt_params is None:
                prompt_params = {}

            if not self._disable_tracer:
                # Create a new span for this guard call
                self._hub_telemetry.create_new_span(
                    span_name="/guard_call",
                    attributes=[
                        ("guard_id", self._guard_id),
                        ("user_id", self._user_id),
                        ("llm_api", llm_api.__name__ if llm_api else "None"),
                        ("custom_reask_prompt", self.reask_prompt is not None),
                        (
                            "custom_reask_instructions",
                            self.reask_instructions is not None,
                        ),
                    ],
                    is_parent=True,  # It will have children
                    has_parent=False,  # Has no parents
                )

            set_call_kwargs(kwargs)
            set_tracer(self._tracer)
            set_tracer_context(self._tracer_context)

            self.configure(num_reasks)
            if self.num_reasks is None:
                raise RuntimeError(
                    "`num_reasks` is `None` after calling `configure()`. "
                    "This should never happen."
                )

            input_prompt = prompt or (self.prompt._source if self.prompt else None)
            input_instructions = instructions or (
                self.instructions._source if self.instructions else None
            )
            call_inputs = CallInputs(
                llm_api=llm_api,
                prompt=input_prompt,
                instructions=input_instructions,
                msg_history=msg_history,
                prompt_params=prompt_params,
                num_reasks=self.num_reasks,
                metadata=metadata,
                full_schema_reask=full_schema_reask,
                args=list(args),
                kwargs=kwargs,
            )
            call_log = Call(inputs=call_inputs)
            set_scope(str(id(call_log)))
            self.history.push(call_log)

            if self._api_client is not None and model_is_supported_server_side(
                llm_api, *args, **kwargs
            ):
                return self._call_server(
                    llm_api=llm_api,
                    num_reasks=self.num_reasks,
                    prompt_params=prompt_params,
                    full_schema_reask=full_schema_reask,
                    call_log=call_log,
                    *args,
                    **kwargs,
                )

            # If the LLM API is not async, fail
            # FIXME: it seems like this check isn't actually working?
            if not inspect.isawaitable(llm_api) and not inspect.iscoroutinefunction(
                llm_api
            ):
                raise RuntimeError(
                    f"The LLM API `{llm_api.__name__}` is not a coroutine. "
                    "Please use an async LLM API."
                )
            # Otherwise, call the LLM
            return await self._call_async(
                llm_api,
                prompt_params=prompt_params,
                num_reasks=self.num_reasks,
                prompt=prompt,
                instructions=instructions,
                msg_history=msg_history,
                metadata=metadata,
                full_schema_reask=full_schema_reask,
                call_log=call_log,
                *args,
                **kwargs,
            )

        guard_context = contextvars.Context()
        return await guard_context.run(
            __call,
            self,
            llm_api,
            prompt_params,
            num_reasks,
            prompt,
            instructions,
            msg_history,
            metadata,
            full_schema_reask,
            *args,
            **kwargs,
        )

    async def _call_async(
        self,
        llm_api: Callable[[Any], Awaitable[Any]],
        prompt_params: Dict,
        num_reasks: int,
        prompt: Optional[str],
        instructions: Optional[str],
        msg_history: Optional[List[Dict]],
        metadata: Dict,
        full_schema_reask: bool,
        call_log: Call,
        *args,
        **kwargs,
    ) -> ValidationOutcome[OT]:
        """Call the LLM asynchronously and validate the output.

        Args:
            llm_api: The LLM API to call asynchronously (e.g. openai.Completion.acreate)
            prompt_params: The parameters to pass to the prompt.format() method.
            num_reasks: The max times to re-ask the LLM for invalid output.
            prompt: The prompt to use for the LLM.
            instructions: Instructions for chat models.
            msg_history: The message history to pass to the LLM.
            metadata: Metadata to pass to the validators.
            full_schema_reask: When reasking, whether to regenerate the full schema
                               or just the incorrect values.
                               Defaults to `True` if a base model is provided,
                               `False` otherwise.

        Returns:
            The raw text output from the LLM and the validated output.
        """
        instructions_obj = instructions or self.rail.instructions
        prompt_obj = prompt or self.rail.prompt
        msg_history_obj = msg_history or []
        if prompt_obj is None:
            if msg_history_obj is not None and not len(msg_history_obj):
                raise RuntimeError(
                    "You must provide a prompt if msg_history is empty. "
                    "Alternatively, you can provide a prompt in the RAIL spec."
                )

        runner = AsyncRunner(
            instructions=instructions_obj,
            prompt=prompt_obj,
            msg_history=msg_history_obj,
            api=get_async_llm_ask(llm_api, *args, **kwargs),
            prompt_schema=self.rail.prompt_schema,
            instructions_schema=self.rail.instructions_schema,
            msg_history_schema=self.rail.msg_history_schema,
            output_schema=self.rail.output_schema,
            num_reasks=num_reasks,
            metadata=metadata,
            base_model=self.base_model,
            full_schema_reask=full_schema_reask,
            disable_tracer=self._disable_tracer,
        )
        call = await runner.async_run(call_log=call_log, prompt_params=prompt_params)
        return ValidationOutcome[OT].from_guard_history(call)

    async def parse(
        self,
        llm_output: str,
        metadata: Optional[Dict] = None,
        llm_api: Optional[Callable] = None,
        num_reasks: Optional[int] = None,
        prompt_params: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> Union[ValidationOutcome[OT], Awaitable[ValidationOutcome[OT]]]:
        """Alternate flow to using Guard where the llm_output is known.

        Args:
            llm_output: The output being parsed and validated.
            metadata: Metadata to pass to the validators.
            llm_api: The LLM API to call
                     (e.g. openai.Completion.create or openai.Completion.acreate)
            num_reasks: The max times to re-ask the LLM for invalid output.
            prompt_params: The parameters to pass to the prompt.format() method.
            full_schema_reask: When reasking, whether to regenerate the full schema
                               or just the incorrect values.

        Returns:
            The validated response. This is either a string or a dictionary,
                determined by the object schema defined in the RAILspec.
        """

        async def __parse(
            self,
            llm_output: str,
            metadata: Optional[Dict] = None,
            llm_api: Optional[Callable] = None,
            num_reasks: Optional[int] = None,
            prompt_params: Optional[Dict] = None,
            full_schema_reask: Optional[bool] = None,
            *args,
            **kwargs,
        ):
            final_num_reasks = (
                num_reasks if num_reasks is not None else 0 if llm_api is None else None
            )

            if not self._disable_tracer:
                self._hub_telemetry.create_new_span(
                    span_name="/guard_parse",
                    attributes=[
                        ("guard_id", self._guard_id),
                        ("user_id", self._user_id),
                        ("llm_api", llm_api.__name__ if llm_api else "None"),
                        ("custom_reask_prompt", self.reask_prompt is not None),
                        (
                            "custom_reask_instructions",
                            self.reask_instructions is not None,
                        ),
                    ],
                    is_parent=True,  # It will have children
                    has_parent=False,  # Has no parents
                )

            self.configure(final_num_reasks)
            if self.num_reasks is None:
                raise RuntimeError(
                    "`num_reasks` is `None` after calling `configure()`. "
                    "This should never happen."
                )
            if full_schema_reask is None:
                full_schema_reask = self.base_model is not None
            metadata = metadata or {}
            prompt_params = prompt_params or {}

            set_call_kwargs(kwargs)
            set_tracer(self._tracer)
            set_tracer_context(self._tracer_context)

            input_prompt = self.prompt._source if self.prompt else None
            input_instructions = (
                self.instructions._source if self.instructions else None
            )
            call_inputs = CallInputs(
                llm_api=llm_api,
                llm_output=llm_output,
                prompt=input_prompt,
                instructions=input_instructions,
                prompt_params=prompt_params,
                num_reasks=self.num_reasks,
                metadata=metadata,
                full_schema_reask=full_schema_reask,
                args=list(args),
                kwargs=kwargs,
            )
            call_log = Call(inputs=call_inputs)
            set_scope(str(id(call_log)))
            self.history.push(call_log)

            if self._api_client is not None and model_is_supported_server_side(
                llm_api, *args, **kwargs
            ):
                return self._call_server(
                    llm_output=llm_output,
                    llm_api=llm_api,
                    num_reasks=self.num_reasks,
                    prompt_params=prompt_params,
                    full_schema_reask=full_schema_reask,
                    call_log=call_log,
                    *args,
                    **kwargs,
                )

            # FIXME: checking not llm_api because it can still fall back on defaults and
            # function as expected. We should handle this better.
            if (
                not llm_api
                or inspect.iscoroutinefunction(llm_api)
                or inspect.isasyncgenfunction(llm_api)
            ):
                return await self._async_parse(
                    llm_output,
                    metadata,
                    llm_api=llm_api,
                    num_reasks=self.num_reasks,
                    prompt_params=prompt_params,
                    full_schema_reask=full_schema_reask,
                    call_log=call_log,
                    *args,
                    **kwargs,
                )

            else:
                raise NotImplementedError(
                    "AsyncGuard does not support non-async LLM APIs. "
                    "Please use the synchronous API Guard or supply an asynchronous "
                    "LLM API."
                )

        guard_context = contextvars.Context()
        return await guard_context.run(
            __parse,
            self,
            llm_output,
            metadata,
            llm_api,
            num_reasks,
            prompt_params,
            full_schema_reask,
            *args,
            **kwargs,
        )

    async def _async_parse(
        self,
        llm_output: str,
        metadata: Dict,
        llm_api: Optional[Callable[[Any], Awaitable[Any]]],
        num_reasks: int,
        prompt_params: Dict,
        full_schema_reask: bool,
        call_log: Call,
        *args,
        **kwargs,
    ) -> ValidationOutcome[OT]:
        """Alternate flow to using Guard where the llm_output is known.

        Args:
            llm_output: The output from the LLM.
            llm_api: The LLM API to use to re-ask the LLM.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            The validated response.
        """
        runner = AsyncRunner(
            instructions=kwargs.pop("instructions", None),
            prompt=kwargs.pop("prompt", None),
            msg_history=kwargs.pop("msg_history", None),
            api=get_async_llm_ask(llm_api, *args, **kwargs) if llm_api else None,
            prompt_schema=self.rail.prompt_schema,
            instructions_schema=self.rail.instructions_schema,
            msg_history_schema=self.rail.msg_history_schema,
            output_schema=self.rail.output_schema,
            num_reasks=num_reasks,
            metadata=metadata,
            output=llm_output,
            base_model=self.base_model,
            full_schema_reask=full_schema_reask,
            disable_tracer=self._disable_tracer,
        )
        call = await runner.async_run(call_log=call_log, prompt_params=prompt_params)

        return ValidationOutcome[OT].from_guard_history(call)
