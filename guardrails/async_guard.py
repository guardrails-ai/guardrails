from builtins import id as object_id
import contextvars
import inspect
import warnings
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    cast,
)

from guardrails_api_client.models import ValidatePayload

from guardrails import Guard
from guardrails.classes import OT, ValidationOutcome
from guardrails.classes.history import Call
from guardrails.classes.history.call_inputs import CallInputs
from guardrails.llm_providers import get_async_llm_ask, model_is_supported_server_side
from guardrails.logger import set_scope
from guardrails.run import AsyncRunner, AsyncStreamRunner
from guardrails.stores.context import (
    get_call_kwarg,
    set_call_kwargs,
    set_tracer,
    set_tracer_context,
)
from guardrails.utils.validator_utils import verify_metadata_requirements


class AsyncGuard(Guard):
    """The AsyncGuard class.

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

    async def _execute(  # FIXME: Is this override necessary?
        self,
        *args,
        llm_api: Optional[Union[Callable, Callable[[Any], Awaitable[Any]]]] = None,
        llm_output: Optional[str] = None,
        prompt_params: Optional[Dict] = None,
        num_reasks: Optional[int] = None,
        messages: Optional[List[Dict]] = None,
        metadata: Optional[Dict],
        full_schema_reask: Optional[bool] = None,
        **kwargs,
    ) -> Union[
        Union[ValidationOutcome[OT], Iterable[ValidationOutcome[OT]]],
        Awaitable[ValidationOutcome[OT]],
    ]:
        self._fill_validator_map()
        self._fill_validators()
        metadata = metadata or {}
        if not llm_output:
            raise RuntimeError("'llm_output' must be provided!")
        if not llm_output and not (messages):
            raise RuntimeError(
                "'messages' must be provided in order to call an LLM!"
            )

        # check if validator requirements are fulfilled
        missing_keys = verify_metadata_requirements(metadata, self._validators)
        if missing_keys:
            raise ValueError(
                f"Missing required metadata keys: {', '.join(missing_keys)}"
            )

        async def __exec(
            self: AsyncGuard,
            *args,
            llm_api: Optional[Union[Callable, Callable[[Any], Awaitable[Any]]]],
            llm_output: Optional[str] = None,
            prompt_params: Optional[Dict] = None,
            num_reasks: Optional[int] = None,
            messages: Optional[List[Dict]] = None,
            metadata: Optional[Dict] = None,
            full_schema_reask: Optional[bool] = None,
            **kwargs,
        ):
            prompt_params = prompt_params or {}
            metadata = metadata or {}
            if full_schema_reask is None:
                full_schema_reask = self._base_model is not None

            if self._allow_metrics_collection:
                # Create a new span for this guard call
                self._hub_telemetry.create_new_span(
                    span_name="/guard_call",
                    attributes=[
                        ("guard_id", self.id),
                        ("user_id", self._user_id),
                        ("llm_api", llm_api.__name__ if llm_api else "None"),
                        (
                            "custom_reask_messages",
                            self._exec_opts.reask_messages is not None,
                        ),
                    ],
                    is_parent=True,  # It will have children
                    has_parent=False,  # Has no parents
                )

            set_call_kwargs(kwargs)
            set_tracer(self._tracer)
            set_tracer_context(self._tracer_context)

            self._set_num_reasks(num_reasks=num_reasks)
            if self._num_reasks is None:
                raise RuntimeError(
                    "`num_reasks` is `None` after calling `configure()`. "
                    "This should never happen."
                )

            input_messages = messages or self._exec_opts.messages
            call_inputs = CallInputs(
                llm_api=llm_api,
                messages=input_messages,
                prompt_params=prompt_params,
                num_reasks=self._num_reasks,
                metadata=metadata,
                full_schema_reask=full_schema_reask,
                args=list(args),
                kwargs=kwargs,
            )
            call_log = Call(inputs=call_inputs)
            set_scope(str(object_id(call_log)))
            self._history_push(call_log)

            if self._api_client is not None and model_is_supported_server_side(
                llm_api, *args, **kwargs
            ):
                result = self._call_server(
                    llm_output=llm_output,
                    llm_api=llm_api,
                    num_reasks=self._num_reasks,
                    prompt_params=prompt_params,
                    metadata=metadata,
                    full_schema_reask=full_schema_reask,
                    call_log=call_log,
                    *args,
                    **kwargs,
                )

            # If the LLM API is async, return a coroutine
            else:
                result = self._exec_async(
                    llm_api=llm_api,
                    llm_output=llm_output,
                    prompt_params=prompt_params,
                    num_reasks=self._num_reasks,
                    messages=messages,
                    metadata=metadata,
                    full_schema_reask=full_schema_reask,
                    call_log=call_log,
                    *args,
                    **kwargs,
                )

            if inspect.isawaitable(result):
                return await result

        guard_context = contextvars.Context()
        return await guard_context.run(
            __exec,
            self,
            llm_api=llm_api,
            llm_output=llm_output,
            prompt_params=prompt_params,
            num_reasks=num_reasks,
            messages=messages,
            metadata=metadata,
            full_schema_reask=full_schema_reask,
            *args,
            **kwargs,
        )

    async def _exec_async(
        self,
        *args,
        llm_api: Optional[Callable[[Any], Awaitable[Any]]],
        llm_output: Optional[str] = None,
        call_log: Call,
        prompt_params: Dict,  # Should be defined at this point
        num_reasks: int = 0,  # Should be defined at this point
        metadata: Dict,  # Should be defined at this point
        full_schema_reask: bool = False,  # Should be defined at this point
        messages: Optional[List[Dict]],
        **kwargs,
    ) -> Union[Awaitable[ValidationOutcome[OT]], AsyncIterable[ValidationOutcome[OT]]]:
        """Call the LLM asynchronously and validate the output.

        Args:
            llm_api: The LLM API to call asynchronously (e.g. openai.Completion.acreate)
            prompt_params: The parameters to pass to the prompt.format() method.
            num_reasks: The max times to re-ask the LLM for invalid output.
            prompt: The prompt to use for the LLM.
            messages: The message history to pass to the LLM.
            metadata: Metadata to pass to the validators.
            full_schema_reask: When reasking, whether to regenerate the full schema
                               or just the incorrect values.
                               Defaults to `True` if a base model is provided,
                               `False` otherwise.

        Returns:
            The raw text output from the LLM and the validated output.
        """
        api = (
            get_async_llm_ask(llm_api, *args, **kwargs) if llm_api is not None else None
        )
        if kwargs.get("stream", False):
            # FIXME
            runner = AsyncStreamRunner(
                output_type=self._output_type,
                output_schema=self.output_schema.to_dict(),
                num_reasks=num_reasks,
                validation_map=self._validator_map,
                messages=messages,
                api=api,
                metadata=metadata,
                output=llm_output,
                base_model=self._base_model,
                full_schema_reask=full_schema_reask,
                disable_tracer=(not self._allow_metrics_collection),
                exec_options=self._exec_opts,
            )
            # Here we have an async generator
            async_generator = runner.async_run(
                call_log=call_log, prompt_params=prompt_params
            )
            return async_generator
        else:
            runner = AsyncRunner(
                output_type=self._output_type,
                output_schema=self.output_schema.to_dict(),
                num_reasks=num_reasks,
                validation_map=self._validator_map,
                messages=messages,
                api=api,
                metadata=metadata,
                output=llm_output,
                base_model=self._base_model,
                full_schema_reask=full_schema_reask,
                disable_tracer=(not self._allow_metrics_collection),
                exec_options=self._exec_opts,
            )
            # Why are we using a different method here instead of just overriding?
            call = await runner.async_run(
                call_log=call_log, prompt_params=prompt_params
            )
            return ValidationOutcome[OT].from_guard_history(call)

    async def __call__(
        self,
        llm_api: Optional[Union[Callable, Callable[[Any], Awaitable[Any]]]],
        *args,
        prompt_params: Optional[Dict] = None,
        num_reasks: Optional[int] = 1,
        messages: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
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
            messages: The message history to pass to the LLM.
            metadata: Metadata to pass to the validators.
            full_schema_reask: When reasking, whether to regenerate the full schema
                               or just the incorrect values.
                               Defaults to `True` if a base model is provided,
                               `False` otherwise.

        Returns:
            The raw text output from the LLM and the validated output.
        """

        messages = messages or []
        if messages is not None and not len(messages):
            raise RuntimeError(
                "You must provide messages. "
                "Alternatively, you can provide a prompt in the Schema constructor."
            )

        return await self._execute(
            *args,
            llm_api=llm_api,
            prompt_params=prompt_params,
            num_reasks=num_reasks,
            messages=messages,
            metadata=metadata,
            full_schema_reask=full_schema_reask,
            **kwargs,
        )

    async def parse(
        self,
        llm_output: str,
        *args,
        metadata: Optional[Dict] = None,
        llm_api: Optional[Callable] = None,
        num_reasks: Optional[int] = None,
        prompt_params: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
        **kwargs,
    ) -> Union[ValidationOutcome[OT], Awaitable[ValidationOutcome[OT]]]:
        """Alternate flow to using AsyncGuard where the llm_output is known.

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

        final_num_reasks = (
            num_reasks
            if num_reasks is not None
            else self._num_reasks
            if self._num_reasks is not None
            else 0
            if llm_api is None
            else 1
        )

        default_messages = self._exec_opts.messages if llm_api else None
        messages = kwargs.pop("messages", default_messages)

        return await self._execute(
            *args,
            llm_output=llm_output,
            llm_api=llm_api,
            prompt_params=prompt_params,
            num_reasks=final_num_reasks,
            messages=messages,
            metadata=metadata,
            full_schema_reask=full_schema_reask,
            **kwargs,
        )

    async def _stream_server_call(
        self,
        *,
        payload: Dict[str, Any],
        llm_output: Optional[str] = None,
        num_reasks: Optional[int] = None,
        prompt_params: Optional[Dict] = None,
        metadata: Optional[Dict] = {},
        full_schema_reask: Optional[bool] = True,
        call_log: Optional[Call],
    ) -> AsyncIterable[ValidationOutcome[OT]]:
        # TODO: Once server side supports async streaming, this function will need to
        # yield async generators, not generators
        if self._api_client:
            validation_output: Optional[Any] = None
            response = self._api_client.stream_validate(
                guard=self,  # type: ignore
                payload=ValidatePayload.from_dict(payload),
                openai_api_key=get_call_kwarg("api_key"),
            )
            for fragment in response:
                validation_output = fragment
                if not validation_output:
                    yield ValidationOutcome[OT](
                        raw_llm_output=None,
                        validated_output=None,
                        validation_passed=False,
                        error="The response from the server was empty!",
                    )
                yield ValidationOutcome[OT](
                    raw_llm_output=validation_output.raw_llm_response,  # type: ignore
                    validated_output=cast(OT, validation_output.validated_output),
                    validation_passed=validation_output.result,
                )
            if validation_output:
                self._construct_history_from_server_response(
                    validation_output=validation_output,
                    llm_output=llm_output,
                    num_reasks=num_reasks,
                    prompt_params=prompt_params,
                    metadata=metadata,
                    full_schema_reask=full_schema_reask,
                    call_log=call_log,
                )
        else:
            raise ValueError("AsyncGuard does not have an api client!")

    async def validate(
        self, llm_output: str, *args, **kwargs
    ) -> Awaitable[ValidationOutcome[OT]]:
        return await self.parse(llm_output=llm_output, *args, **kwargs)
