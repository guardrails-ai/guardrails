from builtins import id as object_id
import contextvars
import inspect
from opentelemetry import context as otel_context
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

from guardrails_api_client.models import (
    ValidatePayload,
    ValidationOutcome as IValidationOutcome,
)

from guardrails import Guard
from guardrails.classes import OT, ValidationOutcome
from guardrails.classes.history import Call
from guardrails.classes.history.call_inputs import CallInputs
from guardrails.classes.output_type import OutputTypes
from guardrails.classes.schema.processed_schema import ProcessedSchema
from guardrails.llm_providers import get_async_llm_ask, model_is_supported_server_side
from guardrails.logger import set_scope
from guardrails.run import AsyncRunner, AsyncStreamRunner
from guardrails.stores.context import (
    Tracer,
    get_call_kwarg,
    set_call_kwargs,
    set_tracer,
    set_tracer_context,
)
from guardrails.types.pydantic import ModelOrListOfModels
from guardrails.types.validator import UseManyValidatorSpec, UseValidatorSpec
from guardrails.utils.telemetry_utils import wrap_with_otel_context
from guardrails.utils.validator_utils import verify_metadata_requirements
from guardrails.validator_base import Validator


class AsyncGuard(Guard, Generic[OT]):
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

    @classmethod
    def _from_rail_schema(
        cls,
        schema: ProcessedSchema,
        rail: str,
        *,
        num_reasks: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        guard = super()._from_rail_schema(
            schema,
            rail,
            num_reasks=num_reasks,
            tracer=tracer,
            name=name,
            description=description,
        )
        if schema.output_type == OutputTypes.STRING:
            return cast(AsyncGuard[str], guard)
        elif schema.output_type == OutputTypes.LIST:
            return cast(AsyncGuard[List], guard)
        else:
            return cast(AsyncGuard[Dict], guard)

    @classmethod
    def from_pydantic(
        cls,
        output_class: ModelOrListOfModels,
        *,
        prompt: Optional[str] = None,  # deprecate this too
        instructions: Optional[str] = None,  # deprecate this too
        num_reasks: Optional[int] = None,
        reask_prompt: Optional[str] = None,  # deprecate this too
        reask_instructions: Optional[str] = None,  # deprecate this too
        tracer: Optional[Tracer] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        guard = super().from_pydantic(
            output_class,
            prompt=prompt,
            instructions=instructions,
            num_reasks=num_reasks,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
            tracer=tracer,
            name=name,
            description=description,
        )
        if guard._output_type == OutputTypes.LIST:
            return cast(AsyncGuard[List], guard)
        else:
            return cast(AsyncGuard[Dict], guard)

    @classmethod
    def from_string(
        cls,
        validators: Sequence[Validator],
        *,
        string_description: Optional[str] = None,
        prompt: Optional[str] = None,  # deprecate this too
        instructions: Optional[str] = None,  # deprecate this too
        reask_prompt: Optional[str] = None,  # deprecate this too
        reask_instructions: Optional[str] = None,  # deprecate this too
        num_reasks: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        guard = super().from_string(
            validators,
            string_description=string_description,
            prompt=prompt,
            instructions=instructions,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
            num_reasks=num_reasks,
            tracer=tracer,
            name=name,
            description=description,
        )
        return cast(AsyncGuard[str], guard)

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional["AsyncGuard"]:
        guard = super().from_dict(obj)
        return cast(AsyncGuard, guard)

    def use(
        self,
        validator: UseValidatorSpec,
        *args,
        on: str = "output",
        **kwargs,
    ) -> "AsyncGuard":
        guard = super().use(validator, *args, on=on, **kwargs)
        return cast(AsyncGuard, guard)

    def use_many(
        self,
        *validators: UseManyValidatorSpec,
        on: str = "output",
    ) -> "AsyncGuard":
        guard = super().use_many(*validators, on=on)  # type: ignore
        return cast(AsyncGuard, guard)

    async def _execute(
        self,
        *args,
        llm_api: Optional[Callable[..., Awaitable[Any]]] = None,
        llm_output: Optional[str] = None,
        prompt_params: Optional[Dict] = None,
        num_reasks: Optional[int] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        metadata: Optional[Dict],
        full_schema_reask: Optional[bool] = None,
        **kwargs,
    ) -> Union[
        ValidationOutcome[OT],
        Awaitable[ValidationOutcome[OT]],
        AsyncIterable[ValidationOutcome[OT]],
    ]:
        self._fill_validator_map()
        self._fill_validators()
        metadata = metadata or {}
        if not llm_output and llm_api and not (prompt or msg_history):
            raise RuntimeError(
                "'prompt' or 'msg_history' must be provided in order to call an LLM!"
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
            llm_api: Optional[Callable[..., Awaitable[Any]]],
            llm_output: Optional[str] = None,
            prompt_params: Optional[Dict] = None,
            num_reasks: Optional[int] = None,
            prompt: Optional[str] = None,
            instructions: Optional[str] = None,
            msg_history: Optional[List[Dict]] = None,
            metadata: Optional[Dict] = None,
            full_schema_reask: Optional[bool] = None,
            **kwargs,
        ) -> Union[
            ValidationOutcome[OT],
            Awaitable[ValidationOutcome[OT]],
            AsyncIterable[ValidationOutcome[OT]],
        ]:
            prompt_params = prompt_params or {}
            metadata = metadata or {}
            if full_schema_reask is None:
                full_schema_reask = self._base_model is not None

            if self._allow_metrics_collection:
                llm_api_str = ""
                if llm_api:
                    llm_api_module_name = (
                        llm_api.__module__ if hasattr(llm_api, "__module__") else ""
                    )
                    llm_api_name = (
                        llm_api.__name__
                        if hasattr(llm_api, "__name__")
                        else type(llm_api).__name__
                    )
                    llm_api_str = f"{llm_api_module_name}.{llm_api_name}"
                # Create a new span for this guard call
                self._hub_telemetry.create_new_span(
                    span_name="/guard_call",
                    attributes=[
                        ("guard_id", self.id),
                        ("user_id", self._user_id),
                        ("llm_api", llm_api_str),
                        (
                            "custom_reask_prompt",
                            self._exec_opts.reask_prompt is not None,
                        ),
                        (
                            "custom_reask_instructions",
                            self._exec_opts.reask_instructions is not None,
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

            input_prompt = prompt or self._exec_opts.prompt
            input_instructions = instructions or self._exec_opts.instructions
            call_inputs = CallInputs(
                llm_api=llm_api,
                prompt=input_prompt,
                instructions=input_instructions,
                msg_history=msg_history,
                prompt_params=prompt_params,
                num_reasks=self._num_reasks,
                metadata=metadata,
                full_schema_reask=full_schema_reask,
                args=list(args),
                kwargs=kwargs,
            )

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
                    prompt=prompt,
                    instructions=instructions,
                    msg_history=msg_history,
                    *args,
                    **kwargs,
                )

            # If the LLM API is async, return a coroutine
            else:
                call_log = Call(inputs=call_inputs)
                set_scope(str(object_id(call_log)))
                self.history.push(call_log)
                result = await self._exec(
                    llm_api=llm_api,
                    llm_output=llm_output,
                    prompt_params=prompt_params,
                    num_reasks=self._num_reasks,
                    prompt=prompt,
                    instructions=instructions,
                    msg_history=msg_history,
                    metadata=metadata,
                    full_schema_reask=full_schema_reask,
                    call_log=call_log,
                    *args,
                    **kwargs,
                )

            if inspect.isawaitable(result):
                return await result
            # TODO: Fix types once async streaming is implemented on server
            return result  # type: ignore

        guard_context = contextvars.Context()
        # get the current otel context and wrap the subsequent call
        #   to preserve otel context if guard call is being called by another
        # framework upstream
        current_otel_context = otel_context.get_current()
        wrapped__exec = wrap_with_otel_context(current_otel_context, __exec)
        return await guard_context.run(
            wrapped__exec,
            self,
            llm_api=llm_api,
            llm_output=llm_output,
            prompt_params=prompt_params,
            num_reasks=num_reasks,
            prompt=prompt,
            instructions=instructions,
            msg_history=msg_history,
            metadata=metadata,
            full_schema_reask=full_schema_reask,
            *args,
            **kwargs,
        )

    async def _exec(
        self,
        *args,
        llm_api: Optional[Callable[[Any], Awaitable[Any]]],
        llm_output: Optional[str] = None,
        call_log: Call,
        prompt_params: Dict,  # Should be defined at this point
        num_reasks: int = 0,  # Should be defined at this point
        metadata: Dict,  # Should be defined at this point
        full_schema_reask: bool = False,  # Should be defined at this point
        prompt: Optional[str],
        instructions: Optional[str],
        msg_history: Optional[List[Dict]],
        **kwargs,
    ) -> Union[
        ValidationOutcome[OT],
        Awaitable[ValidationOutcome[OT]],
        AsyncIterable[ValidationOutcome[OT]],
    ]:
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
        api = get_async_llm_ask(llm_api, *args, **kwargs)  # type: ignore
        if kwargs.get("stream", False):
            runner = AsyncStreamRunner(
                output_type=self._output_type,
                output_schema=self.output_schema.to_dict(),
                num_reasks=num_reasks,
                validation_map=self._validator_map,
                prompt=prompt,
                instructions=instructions,
                msg_history=msg_history,
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
                prompt=prompt,
                instructions=instructions,
                msg_history=msg_history,
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
        llm_api: Optional[Callable[..., Awaitable[Any]]] = None,
        *args,
        prompt_params: Optional[Dict] = None,
        num_reasks: Optional[int] = 1,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
        **kwargs,
    ) -> Union[
        ValidationOutcome[OT],
        Awaitable[ValidationOutcome[OT]],
        AsyncIterable[ValidationOutcome[OT]],
    ]:
        """Call the LLM and validate the output. Pass an async LLM API to
        return a coroutine.

        Args:
            llm_api: The LLM API to call
                     (e.g. openai.completions.create or openai.chat.completions.create)
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

        instructions = instructions or self._exec_opts.instructions
        prompt = prompt or self._exec_opts.prompt
        msg_history = msg_history or kwargs.pop("messages", None) or []

        if prompt is None:
            if msg_history is not None and not len(msg_history):
                raise RuntimeError(
                    "You must provide a prompt if msg_history is empty. "
                    "Alternatively, you can provide a prompt in the Schema constructor."
                )

        return await self._execute(
            *args,
            llm_api=llm_api,
            prompt_params=prompt_params,
            num_reasks=num_reasks,
            prompt=prompt,
            instructions=instructions,
            msg_history=msg_history,
            metadata=metadata,
            full_schema_reask=full_schema_reask,
            **kwargs,
        )

    async def parse(
        self,
        llm_output: str,
        *args,
        metadata: Optional[Dict] = None,
        llm_api: Optional[Callable[..., Awaitable[Any]]] = None,
        num_reasks: Optional[int] = None,
        prompt_params: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
        **kwargs,
    ) -> Awaitable[ValidationOutcome[OT]]:
        """Alternate flow to using AsyncGuard where the llm_output is known.

        Args:
            llm_output: The output being parsed and validated.
            metadata: Metadata to pass to the validators.
            llm_api: The LLM API to call
                     (e.g. openai.completions.create or openai.Completion.acreate)
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
        default_prompt = self._exec_opts.prompt if llm_api is not None else None
        prompt = kwargs.pop("prompt", default_prompt)

        default_instructions = self._exec_opts.instructions if llm_api else None
        instructions = kwargs.pop("instructions", default_instructions)

        default_msg_history = self._exec_opts.msg_history if llm_api else None
        msg_history = kwargs.pop("msg_history", default_msg_history)

        return await self._execute(  # type: ignore
            *args,
            llm_output=llm_output,
            llm_api=llm_api,
            prompt_params=prompt_params,
            num_reasks=final_num_reasks,
            prompt=prompt,
            instructions=instructions,
            msg_history=msg_history,
            metadata=metadata,
            full_schema_reask=full_schema_reask,
            **kwargs,
        )

    async def _stream_server_call(
        self, *, payload: Dict[str, Any]
    ) -> AsyncIterable[ValidationOutcome[OT]]:
        # TODO: Once server side supports async streaming, this function will need to
        # yield async generators, not generators
        if self._api_client:
            validation_output: Optional[IValidationOutcome] = None
            response = self._api_client.stream_validate(
                guard=self,  # type: ignore
                payload=ValidatePayload.from_dict(payload),  # type: ignore
                openai_api_key=get_call_kwarg("api_key"),
            )
            for fragment in response:
                validation_output = fragment
                if validation_output is None:
                    yield ValidationOutcome[OT](
                        call_id="0",  # type: ignore
                        raw_llm_output=None,
                        validated_output=None,
                        validation_passed=False,
                        error="The response from the server was empty!",
                    )
                else:
                    validated_output = (
                        cast(OT, validation_output.validated_output.actual_instance)
                        if validation_output.validated_output
                        else None
                    )
                    yield ValidationOutcome[OT](
                        call_id=validation_output.call_id,  # type: ignore
                        raw_llm_output=validation_output.raw_llm_output,  # type: ignore
                        validated_output=validated_output,
                        validation_passed=(validation_output.validation_passed is True),
                    )
            if validation_output:
                guard_history = self._api_client.get_history(
                    self.name, validation_output.call_id
                )
                self.history.extend(
                    [Call.from_interface(call) for call in guard_history]
                )
        else:
            raise ValueError("AsyncGuard does not have an api client!")

    async def validate(
        self, llm_output: str, *args, **kwargs
    ) -> Awaitable[ValidationOutcome[OT]]:
        return await self.parse(llm_output=llm_output, *args, **kwargs)
