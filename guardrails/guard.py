import asyncio
import contextvars
import json
import os
import warnings
from copy import deepcopy
from string import Template
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)

from guardrails_api_client.models import (
    AnyObject,
    Guard as GuardModel,
    History,
    HistoryEvent,
    ValidatePayload,
    ValidationOutput,
)
from guardrails_api_client.types import UNSET
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel
from pydantic.version import VERSION as PYDANTIC_VERSION
from typing_extensions import deprecated  # type: ignore

from guardrails.api_client import GuardrailsApiClient
from guardrails.classes import OT, InputType, ValidationOutcome
from guardrails.classes.credentials import Credentials
from guardrails.classes.generic import Stack
from guardrails.classes.history import Call
from guardrails.classes.history.call_inputs import CallInputs
from guardrails.classes.history.inputs import Inputs
from guardrails.classes.history.iteration import Iteration
from guardrails.classes.history.outputs import Outputs
from guardrails.errors import ValidationError
from guardrails.llm_providers import (
    get_async_llm_ask,
    get_llm_api_enum,
    get_llm_ask,
    model_is_supported_server_side,
)
from guardrails.logger import logger, set_scope
from guardrails.prompt import Instructions, Prompt
from guardrails.rail import Rail
from guardrails.run import AsyncRunner, Runner, StreamRunner
from guardrails.schema import Schema, StringSchema
from guardrails.stores.context import (
    Tracer,
    get_call_kwarg,
    get_tracer_context,
    set_call_kwargs,
    set_tracer,
    set_tracer_context,
)
from guardrails.utils.hub_telemetry_utils import HubTelemetry
from guardrails.utils.llm_response import LLMResponse
from guardrails.utils.reask_utils import FieldReAsk
from guardrails.utils.validator_utils import get_validator
from guardrails.validator_base import FailResult, Validator


class Guard(Runnable, Generic[OT]):
    """The Guard class.

    This class is the main entry point for using Guardrails. It is
    initialized from one of the following class methods:

    - `from_rail`
    - `from_rail_string`
    - `from_pydantic`
    - `from_string`

    The `__call__`
    method functions as a wrapper around LLM APIs. It takes in an LLM
    API, and optional prompt parameters, and returns the raw output from
    the LLM and the validated output.
    """

    _tracer = None
    _tracer_context = None
    _hub_telemetry = None
    _guard_id = None
    _user_id = None
    _validators: List[Validator]
    _api_client: Optional[GuardrailsApiClient] = None

    def __init__(
        self,
        rail: Optional[Rail] = None,
        num_reasks: Optional[int] = None,
        base_model: Optional[
            Union[Type[BaseModel], Type[List[Type[BaseModel]]]]
        ] = None,
        tracer: Optional[Tracer] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the Guard with optional Rail instance, num_reasks, and
        base_model."""
        if not rail:
            rail = (
                Rail.from_pydantic(base_model)
                if base_model
                else Rail.from_string_validators([])
            )
        self.rail = rail
        self.num_reasks = num_reasks
        # TODO: Support a sink for history so that it is not solely held in memory
        self.history: Stack[Call] = Stack()
        self.base_model = base_model
        self._set_tracer(tracer)

        credentials = Credentials.from_rc_file(logger)

        # Get unique id of user from credentials
        self._user_id = credentials.id or ""

        # Get metrics opt-out from credentials
        self._disable_tracer = not credentials.enable_metrics

        # Get id of guard object (that is unique)
        self._guard_id = id(self)  # id of guard object; not the class

        # Initialize Hub Telemetry singleton and get the tracer
        #  if it is not disabled
        if not self._disable_tracer:
            self._hub_telemetry = HubTelemetry()
        self._validators = []

        # Gaurdrails As A Service Initialization
        self.description = description
        self.name = name

        api_key = os.environ.get("GUARDRAILS_API_KEY")
        if api_key is not None:
            if self.name is None:
                self.name = f"gr-{str(self._guard_id)}"
                logger.warn("Warning: No name passed to guard!")
                logger.warn(
                    "Use this auto-generated name to re-use this guard: {name}".format(
                        name=self.name
                    )
                )
            self._api_client = GuardrailsApiClient(api_key=api_key)
            self.upsert_guard()

    @property
    @deprecated(
        """'Guard.prompt_schema' is deprecated and will be removed in \
versions 0.5.x and beyond."""
    )
    def prompt_schema(self) -> Optional[StringSchema]:
        """Return the input schema."""
        return self.rail.prompt_schema

    @property
    @deprecated(
        """'Guard.instructions_schema' is deprecated and will be removed in \
versions 0.5.x and beyond."""
    )
    def instructions_schema(self) -> Optional[StringSchema]:
        """Return the input schema."""
        return self.rail.instructions_schema

    @property
    @deprecated(
        """'Guard.msg_history_schema' is deprecated and will be removed in \
versions 0.5.x and beyond."""
    )
    def msg_history_schema(self) -> Optional[StringSchema]:
        """Return the input schema."""
        return self.rail.msg_history_schema

    @property
    @deprecated(
        """'Guard.output_schema' is deprecated and will be removed in \
versions 0.5.x and beyond."""
    )
    def output_schema(self) -> Schema:
        """Return the output schema."""
        return self.rail.output_schema

    @property
    @deprecated(
        """'Guard.instructions' is deprecated and will be removed in \
versions 0.5.x and beyond. Use 'Guard.history.last.instructions' instead."""
    )
    def instructions(self) -> Optional[Instructions]:
        """Return the instruction-prompt."""
        return self.rail.instructions

    @property
    @deprecated(
        """'Guard.prompt' is deprecated and will be removed in \
versions 0.5.x and beyond. Use 'Guard.history.last.prompt' instead."""
    )
    def prompt(self) -> Optional[Prompt]:
        """Return the prompt."""
        return self.rail.prompt

    @property
    @deprecated(
        """'Guard.raw_prompt' is deprecated and will be removed in \
versions 0.5.x and beyond. Use 'Guard.history.last.prompt' instead."""
    )
    def raw_prompt(self) -> Optional[Prompt]:
        """Return the prompt, alias for `prompt`."""
        return self.rail.prompt

    @property
    @deprecated(
        """'Guard.base_prompt' is deprecated and will be removed in \
versions 0.5.x and beyond. Use 'Guard.history.last.prompt' instead."""
    )
    def base_prompt(self) -> Optional[str]:
        """Return the base prompt i.e. prompt.source."""
        if self.rail.prompt is None:
            return None
        return self.rail.prompt.source

    @property
    @deprecated(
        """'Guard.reask_prompt' is deprecated and will be removed in \
versions 0.5.x and beyond. Use 'Guard.history.last.reask_prompts' instead."""
    )
    def reask_prompt(self) -> Optional[Prompt]:
        """Return the reask prompt."""
        return self.rail.output_schema.reask_prompt_template

    @reask_prompt.setter
    @deprecated(
        """'Guard.reask_prompt' is deprecated and will be removed in \
versions 0.5.x and beyond. Pass 'reask_prompt' in the initializer \
    method instead: e.g. 'Guard.from_pydantic'."""
    )
    def reask_prompt(self, reask_prompt: Optional[str]):
        """Set the reask prompt."""
        self.rail.output_schema.reask_prompt_template = reask_prompt

    @property
    @deprecated(
        """'Guard.reask_instructions' is deprecated and will be removed in \
versions 0.5.x and beyond. Use 'Guard.history.last.reask_instructions' instead."""
    )
    def reask_instructions(self) -> Optional[Instructions]:
        """Return the reask prompt."""
        return self.rail.output_schema.reask_instructions_template

    @reask_instructions.setter
    @deprecated(
        """'Guard.reask_instructions' is deprecated and will be removed in \
versions 0.5.x and beyond. Pass 'reask_instructions' in the initializer \
    method instead: e.g. 'Guard.from_pydantic'."""
    )
    def reask_instructions(self, reask_instructions: Optional[str]):
        """Set the reask prompt."""
        self.rail.output_schema.reask_instructions_template = reask_instructions

    def configure(
        self,
        num_reasks: Optional[int] = None,
    ):
        """Configure the Guard."""
        self.num_reasks = (
            num_reasks
            if num_reasks is not None
            else self.num_reasks
            if self.num_reasks is not None
            else 1
        )

    def _set_tracer(self, tracer: Optional[Tracer] = None) -> None:
        self._tracer = tracer
        set_tracer(tracer)
        set_tracer_context()
        self._tracer_context = get_tracer_context()

    @classmethod
    def from_rail(
        cls,
        rail_file: str,
        num_reasks: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Create a Schema from a `.rail` file.

        Args:
            rail_file: The path to the `.rail` file.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            An instance of the `Guard` class.
        """

        # We have to set the tracer in the ContextStore before the Rail,
        #   and therefore the Validators, are initialized
        cls._set_tracer(cls, tracer)  # type: ignore

        rail = Rail.from_file(rail_file)
        if rail.output_type == "str":
            return cast(
                Guard[str],
                cls(
                    rail=rail,
                    num_reasks=num_reasks,
                    tracer=tracer,
                    name=name,
                    description=description,
                ),
            )
        elif rail.output_type == "list":
            return cast(
                Guard[List],
                cls(
                    rail=rail,
                    num_reasks=num_reasks,
                    tracer=tracer,
                    name=name,
                    description=description,
                ),
            )
        return cast(
            Guard[Dict],
            cls(
                rail=rail,
                num_reasks=num_reasks,
                tracer=tracer,
                name=name,
                description=description,
            ),
        )

    @classmethod
    def from_rail_string(
        cls,
        rail_string: str,
        num_reasks: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Create a Schema from a `.rail` string.

        Args:
            rail_string: The `.rail` string.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            An instance of the `Guard` class.
        """
        # We have to set the tracer in the ContextStore before the Rail,
        #   and therefore the Validators, are initialized
        cls._set_tracer(cls, tracer)  # type: ignore

        rail = Rail.from_string(rail_string)
        if rail.output_type == "str":
            return cast(
                Guard[str],
                cls(
                    rail=rail,
                    num_reasks=num_reasks,
                    tracer=tracer,
                    name=name,
                    description=description,
                ),
            )
        elif rail.output_type == "list":
            return cast(
                Guard[List],
                cls(
                    rail=rail,
                    num_reasks=num_reasks,
                    tracer=tracer,
                    name=name,
                    description=description,
                ),
            )
        return cast(
            Guard[Dict],
            cls(
                rail=rail,
                num_reasks=num_reasks,
                tracer=tracer,
                name=name,
                description=description,
            ),
        )

    @classmethod
    def from_pydantic(
        cls,
        output_class: Union[Type[BaseModel], Type[List[Type[BaseModel]]]],
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        num_reasks: Optional[int] = None,
        reask_prompt: Optional[str] = None,
        reask_instructions: Optional[str] = None,
        tracer: Optional[Tracer] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Create a Guard instance from a Pydantic model and prompt."""
        if PYDANTIC_VERSION.startswith("1"):
            warnings.warn(
                """Support for Pydantic v1.x is deprecated and will be removed in
                Guardrails 0.5.x. Please upgrade to the latest Pydantic v2.x to
                continue receiving future updates and support.""",
                FutureWarning,
            )
        # We have to set the tracer in the ContextStore before the Rail,
        #   and therefore the Validators, are initialized
        cls._set_tracer(cls, tracer)  # type: ignore

        rail = Rail.from_pydantic(
            output_class=output_class,
            prompt=prompt,
            instructions=instructions,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
        )
        if rail.output_type == "list":
            return cast(
                Guard[List], cls(rail, num_reasks=num_reasks, base_model=output_class)
            )
        return cast(
            Guard[Dict],
            cls(
                rail,
                num_reasks=num_reasks,
                base_model=output_class,
                tracer=tracer,
                name=name,
                description=description,
            ),
        )

    @classmethod
    def from_string(
        cls,
        validators: Sequence[Validator],
        description: Optional[str] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        reask_prompt: Optional[str] = None,
        reask_instructions: Optional[str] = None,
        num_reasks: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        *,
        name: Optional[str] = None,
        guard_description: Optional[str] = None,
    ):
        """Create a Guard instance for a string response with prompt,
        instructions, and validations.

        Args:
            validators: (List[Validator]): The list of validators to apply to the string output.
            description (str, optional): A description for the string to be generated. Defaults to None.
            prompt (str, optional): The prompt used to generate the string. Defaults to None.
            instructions (str, optional): Instructions for chat models. Defaults to None.
            reask_prompt (str, optional): An alternative prompt to use during reasks. Defaults to None.
            reask_instructions (str, optional): Alternative instructions to use during reasks. Defaults to None.
            num_reasks (int, optional): The max times to re-ask the LLM for invalid output.
        """  # noqa

        cls._set_tracer(cls, tracer)  # type: ignore

        rail = Rail.from_string_validators(
            validators=validators,
            description=description,
            prompt=prompt,
            instructions=instructions,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
        )
        return cast(
            Guard[str],
            cls(
                rail,
                num_reasks=num_reasks,
                tracer=tracer,
                name=name,
                description=guard_description,
            ),
        )

    @overload
    def __call__(
        self,
        llm_api: Callable,
        prompt_params: Optional[Dict] = None,
        num_reasks: Optional[int] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
        stream: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Union[ValidationOutcome[OT], Iterable[ValidationOutcome[OT]]]: ...

    @overload
    def __call__(
        self,
        llm_api: Callable[[Any], Awaitable[Any]],
        prompt_params: Optional[Dict] = None,
        num_reasks: Optional[int] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> Awaitable[ValidationOutcome[OT]]: ...

    def __call__(
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
        """Call the LLM and validate the output.

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

        def __call(
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
            llm_api_str = (
                f"{llm_api.__module__}.{llm_api.__name__}" if llm_api else "None"
            )
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
                        ("llm_api", llm_api_str),
                        (
                            "custom_reask_prompt",
                            self.rail.output_schema.reask_prompt_template is not None,
                        ),
                        (
                            "custom_reask_instructions",
                            self.rail.output_schema.reask_instructions_template
                            is not None,
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

            input_prompt = prompt or (
                self.rail.prompt._source if self.rail.prompt else None
            )
            input_instructions = instructions or (
                self.rail.instructions._source if self.rail.instructions else None
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

            # If the LLM API is async, return a coroutine. This will be deprecated soon.

            if asyncio.iscoroutinefunction(llm_api):
                return self._call_async(
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
            # Otherwise, call the LLM synchronously
            return self._call_sync(
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
        return guard_context.run(
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

    def _call_sync(
        self,
        llm_api: Callable,
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
    ) -> Union[ValidationOutcome[OT], Iterable[ValidationOutcome[OT]]]:
        instructions_obj = instructions or self.rail.instructions
        prompt_obj = prompt or self.rail.prompt
        msg_history_obj = msg_history or []
        if prompt_obj is None:
            if msg_history is not None and not len(msg_history_obj):
                raise RuntimeError(
                    "You must provide a prompt if msg_history is empty. "
                    "Alternatively, you can provide a prompt in the Schema constructor."
                )

        # Check whether stream is set
        if kwargs.get("stream", False):
            # If stream is True, use StreamRunner
            runner = StreamRunner(
                instructions=instructions_obj,
                prompt=prompt_obj,
                msg_history=msg_history_obj,
                api=get_llm_ask(llm_api, *args, **kwargs),
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
            return runner(call_log=call_log, prompt_params=prompt_params)
        else:
            # Otherwise, use Runner
            runner = Runner(
                instructions=instructions_obj,
                prompt=prompt_obj,
                msg_history=msg_history_obj,
                api=get_llm_ask(llm_api, *args, **kwargs),
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
            call = runner(call_log=call_log, prompt_params=prompt_params)
            return ValidationOutcome[OT].from_guard_history(call)

    @deprecated(
        """Async methods within Guard are deprecated and will be removed in 0.5.x.
        Instead, please use `AsyncGuard() or pass in a synchronous llm api.""",
        category=FutureWarning,
        stacklevel=2,
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

    def __repr__(self):
        return f"Guard(RAIL={self.rail})"

    def __rich_repr__(self):
        yield "RAIL", self.rail

    def __stringify__(self):
        if self.rail and self.rail.output_type == "str":
            template = Template(
                """
                Guard {
                    validators: [
                        ${validators}
                    ]
                }
                    """
            )
            return template.safe_substitute(
                {
                    "validators": ",\n".join(
                        [v.__stringify__() for v in self._validators]
                    )
                }
            )
        return self.__repr__()

    @overload
    def parse(
        self,
        llm_output: str,
        metadata: Optional[Dict] = None,
        llm_api: None = None,
        num_reasks: Optional[int] = None,
        prompt_params: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> ValidationOutcome[OT]: ...

    @overload
    def parse(
        self,
        llm_output: str,
        metadata: Optional[Dict] = None,
        llm_api: Callable[[Any], Awaitable[Any]] = ...,
        num_reasks: Optional[int] = None,
        prompt_params: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> Awaitable[ValidationOutcome[OT]]: ...

    @overload
    def parse(
        self,
        llm_output: str,
        metadata: Optional[Dict] = None,
        llm_api: Optional[Callable] = None,
        num_reasks: Optional[int] = None,
        prompt_params: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> ValidationOutcome[OT]: ...

    def parse(
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

        def __parse(
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
            llm_api_str = (
                f"{llm_api.__module__}.{llm_api.__name__}" if llm_api else "None"
            )
            final_num_reasks = (
                num_reasks if num_reasks is not None else 0 if llm_api is None else None
            )

            if not self._disable_tracer:
                self._hub_telemetry.create_new_span(
                    span_name="/guard_parse",
                    attributes=[
                        ("guard_id", self._guard_id),
                        ("user_id", self._user_id),
                        ("llm_api", llm_api_str),
                        (
                            "custom_reask_prompt",
                            self.rail.output_schema.reask_prompt_template is not None,
                        ),
                        (
                            "custom_reask_instructions",
                            self.rail.output_schema.reask_instructions_template
                            is not None,
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

            input_prompt = self.rail.prompt._source if self.rail.prompt else None
            input_instructions = (
                self.rail.instructions._source if self.rail.instructions else None
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

            # If the LLM API is async, return a coroutine
            if asyncio.iscoroutinefunction(llm_api):
                return self._async_parse(
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
            # Otherwise, call the LLM synchronously
            return self._sync_parse(
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

        guard_context = contextvars.Context()
        return guard_context.run(
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

    def _sync_parse(
        self,
        llm_output: str,
        metadata: Dict,
        llm_api: Optional[Callable],
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
        runner = Runner(
            instructions=kwargs.pop("instructions", None),
            prompt=kwargs.pop("prompt", None),
            msg_history=kwargs.pop("msg_history", None),
            api=get_llm_ask(llm_api, *args, **kwargs) if llm_api else None,
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
        call = runner(call_log=call_log, prompt_params=prompt_params)

        return ValidationOutcome[OT].from_guard_history(call)

    @deprecated(
        """Async methods within Guard are deprecated and will be removed in 0.5.x.
        Instead, please use `AsyncGuard() or pass in a synchronous llm api.""",
        category=FutureWarning,
        stacklevel=2,
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

    @deprecated(
        """The `with_prompt_validation` method is deprecated,
        and will be removed in 0.5.x. Instead, please use
        `Guard().use(YourValidator, on='prompt')`.""",
        category=FutureWarning,
        stacklevel=2,
    )
    def with_prompt_validation(
        self,
        validators: Sequence[Validator],
    ):
        """Add prompt validation to the Guard.

        Args:
            validators: The validators to add to the prompt.
        """
        if self.rail.prompt_schema:
            warnings.warn("Overriding existing prompt validators.")
        schema = StringSchema.from_string(
            validators=validators,
        )
        self.rail.prompt_schema = schema
        return self

    @deprecated(
        """The `with_instructions_validation` method is deprecated,
        and will be removed in 0.5.x. Instead, please use
        `Guard().use(YourValidator, on='instructions')`.""",
        category=FutureWarning,
        stacklevel=2,
    )
    def with_instructions_validation(
        self,
        validators: Sequence[Validator],
    ):
        """Add instructions validation to the Guard.

        Args:
            validators: The validators to add to the instructions.
        """
        if self.rail.instructions_schema:
            warnings.warn("Overriding existing instructions validators.")
        schema = StringSchema.from_string(
            validators=validators,
        )
        self.rail.instructions_schema = schema
        return self

    @deprecated(
        """The `with_msg_history_validation` method is deprecated,
        and will be removed in 0.5.x. Instead, please use
        `Guard().use(YourValidator, on='msg_history')`.""",
        category=FutureWarning,
        stacklevel=2,
    )
    def with_msg_history_validation(
        self,
        validators: Sequence[Validator],
    ):
        """Add msg_history validation to the Guard.

        Args:
            validators: The validators to add to the msg_history.
        """
        if self.rail.msg_history_schema:
            warnings.warn("Overriding existing msg_history validators.")
        schema = StringSchema.from_string(
            validators=validators,
        )
        self.rail.msg_history_schema = schema
        return self

    def __add_validator(self, validator: Validator, on: str = "output"):
        # Only available for string output types
        if self.rail.output_type != "str":
            raise RuntimeError(
                "The `use` method is only available for string output types."
            )

        if on == "prompt":
            # If the prompt schema exists, add the validator to it
            if self.rail.prompt_schema:
                self.rail.prompt_schema.root_datatype.validators.append(validator)
            else:
                # Otherwise, create a new schema with the validator
                schema = StringSchema.from_string(
                    validators=[validator],
                )
                self.rail.prompt_schema = schema
        elif on == "instructions":
            # If the instructions schema exists, add the validator to it
            if self.rail.instructions_schema:
                self.rail.instructions_schema.root_datatype.validators.append(validator)
            else:
                # Otherwise, create a new schema with the validator
                schema = StringSchema.from_string(
                    validators=[validator],
                )
                self.rail.instructions_schema = schema
        elif on == "msg_history":
            # If the msg_history schema exists, add the validator to it
            if self.rail.msg_history_schema:
                self.rail.msg_history_schema.root_datatype.validators.append(validator)
            else:
                # Otherwise, create a new schema with the validator
                schema = StringSchema.from_string(
                    validators=[validator],
                )
                self.rail.msg_history_schema = schema
        elif on == "output":
            self._validators.append(validator)
            self.rail.output_schema.root_datatype.validators.append(validator)
        else:
            raise ValueError(
                """Invalid value for `on`. Must be one of the following:
                'output', 'prompt', 'instructions', 'msg_history'."""
            )

    @overload
    def use(self, validator: Validator, *, on: str = "output") -> "Guard": ...

    @overload
    def use(
        self, validator: Type[Validator], *args, on: str = "output", **kwargs
    ) -> "Guard": ...

    def use(
        self,
        validator: Union[Validator, Type[Validator]],
        *args,
        on: str = "output",
        **kwargs,
    ) -> "Guard":
        """Use a validator to validate either of the following:
        - The output of an LLM request
        - The prompt
        - The instructions
        - The message history

        *Note*: For on="output", `use` is only available for string output types.

        Args:
            validator: The validator to use. Either the class or an instance.
            on: The part of the LLM request to validate. Defaults to "output".
        """
        hydrated_validator = get_validator(validator, *args, **kwargs)
        self.__add_validator(hydrated_validator, on=on)
        return self

    @overload
    def use_many(self, *validators: Validator, on: str = "output") -> "Guard": ...

    @overload
    def use_many(
        self,
        *validators: Tuple[
            Type[Validator],
            Optional[Union[List[Any], Dict[str, Any]]],
            Optional[Dict[str, Any]],
        ],
        on: str = "output",
    ) -> "Guard": ...

    def use_many(
        self,
        *validators: Union[
            Validator,
            Tuple[
                Type[Validator],
                Optional[Union[List[Any], Dict[str, Any]]],
                Optional[Dict[str, Any]],
            ],
        ],
        on: str = "output",
    ) -> "Guard":
        """Use a validator to validate results of an LLM request.

        *Note*: `use_many` is only available for string output types.
        """
        if self.rail.output_type != "str":
            raise RuntimeError(
                "The `use_many` method is only available for string output types."
            )

        # Loop through the validators
        for v in validators:
            hydrated_validator = get_validator(v)
            self.__add_validator(hydrated_validator, on=on)
        return self

    def validate(self, llm_output: str, *args, **kwargs) -> ValidationOutcome[str]:
        if (
            not self.rail
            or self.rail.output_schema.root_datatype.validators != self._validators
        ):
            self.rail = Rail.from_string_validators(
                validators=self._validators,
                prompt=self.rail.prompt.source if self.rail.prompt else None,
                instructions=(
                    self.rail.instructions.source if self.rail.instructions else None
                ),
                reask_prompt=(
                    self.rail.output_schema.reask_prompt_template.source
                    if self.rail.output_schema.reask_prompt_template
                    else None
                ),
                reask_instructions=self.rail.output_schema.reask_instructions_template.source
                if self.rail.output_schema.reask_instructions_template
                else None,
            )

        return self.parse(llm_output=llm_output, *args, **kwargs)

    # No call support for this until
    # https://github.com/guardrails-ai/guardrails/pull/525 is merged
    # def __call__(self, llm_output: str, *args, **kwargs) -> ValidationOutcome[str]:
    #     return self.validate(llm_output, *args, **kwargs)

    def invoke(
        self, input: InputType, config: Optional[RunnableConfig] = None
    ) -> InputType:
        output = BaseMessage(content="", type="")
        str_input = None
        input_is_chat_message = False
        if isinstance(input, BaseMessage):
            input_is_chat_message = True
            str_input = str(input.content)
            output = deepcopy(input)
        else:
            str_input = str(input)

        response = self.validate(str_input)

        validated_output = response.validated_output
        if not validated_output:
            raise ValidationError(
                (
                    "The response from the LLM failed validation!"
                    "See `guard.history` for more details."
                )
            )

        if isinstance(validated_output, Dict):
            validated_output = json.dumps(validated_output)

        if input_is_chat_message:
            output.content = validated_output
            return cast(InputType, output)
        return cast(InputType, validated_output)

    def _to_request(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "railspec": self.rail._to_request(),
            "numReasks": self.num_reasks,
        }

    def upsert_guard(self):
        if self._api_client:
            guard_dict = self._to_request()
            self._api_client.upsert_guard(GuardModel.from_dict(guard_dict))
        else:
            raise ValueError("Guard does not have an api client!")

    def _call_server(
        self,
        *args,
        llm_output: Optional[str] = None,
        llm_api: Optional[Callable] = None,
        num_reasks: Optional[int] = None,
        prompt_params: Optional[Dict] = None,
        metadata: Optional[Dict] = {},
        full_schema_reask: Optional[bool] = True,
        call_log: Optional[Call],
        # prompt: Optional[str],
        # instructions: Optional[str],
        # msg_history: Optional[List[Dict]],
        **kwargs,
    ):
        if self._api_client:
            payload: Dict[str, Any] = {"args": list(args)}
            payload.update(**kwargs)
            if llm_output is not None:
                payload["llmOutput"] = llm_output
            if num_reasks is not None:
                payload["numReasks"] = num_reasks
            if prompt_params is not None:
                payload["promptParams"] = prompt_params
            if llm_api is not None:
                payload["llmApi"] = get_llm_api_enum(llm_api)
            # TODO: get enum for llm_api
            validation_output: Optional[ValidationOutput] = self._api_client.validate(
                guard=self,  # type: ignore
                payload=ValidatePayload.from_dict(payload),
                openai_api_key=get_call_kwarg("api_key"),
            )

            if not validation_output:
                return ValidationOutcome[OT](
                    raw_llm_output=None,
                    validated_output=None,
                    validation_passed=False,
                    error="The response from the server was empty!",
                )

            call_log = call_log or Call()
            if llm_api is not None:
                llm_api = get_llm_ask(llm_api)
                if asyncio.iscoroutinefunction(llm_api):
                    llm_api = get_async_llm_ask(llm_api)
            session_history = (
                validation_output.session_history
                if validation_output is not None and validation_output.session_history
                else []
            )
            history: History
            for history in session_history:
                history_events: Optional[List[HistoryEvent]] = (  # type: ignore
                    history.history if history.history != UNSET else None
                )
                if history_events is None:
                    continue

                iterations = [
                    Iteration(
                        inputs=Inputs(
                            llm_api=llm_api,
                            llm_output=llm_output,
                            instructions=(
                                Instructions(h.instructions) if h.instructions else None
                            ),
                            prompt=(
                                Prompt(h.prompt.source)  # type: ignore
                                if h.prompt is not None and h.prompt != UNSET
                                else None
                            ),
                            prompt_params=prompt_params,
                            num_reasks=(num_reasks or 0),
                            metadata=metadata,
                            full_schema_reask=full_schema_reask,
                        ),
                        outputs=Outputs(
                            llm_response_info=LLMResponse(
                                output=h.output  # type: ignore
                            ),
                            raw_output=h.output,
                            parsed_output=(
                                h.parsed_output.to_dict()
                                if isinstance(h.parsed_output, AnyObject)
                                else h.parsed_output
                            ),
                            validation_output=(
                                h.validated_output.to_dict()
                                if isinstance(h.validated_output, AnyObject)
                                else h.validated_output
                            ),
                            reasks=list(
                                [
                                    FieldReAsk(
                                        incorrect_value=r.to_dict().get(
                                            "incorrect_value"
                                        ),
                                        path=r.to_dict().get("path"),
                                        fail_results=[
                                            FailResult(
                                                error_message=r.to_dict().get(
                                                    "error_message"
                                                ),
                                                fix_value=r.to_dict().get("fix_value"),
                                            )
                                        ],
                                    )
                                    for r in h.reasks  # type: ignore
                                ]
                                if h.reasks != UNSET
                                else []
                            ),
                        ),
                    )
                    for h in history_events
                ]
                call_log.iterations.extend(iterations)
                if self.history.length == 0:
                    self.history.push(call_log)

            # Our interfaces are too different for this to work right now.
            # Once we move towards shared interfaces for both the open source
            # and the api we can re-enable this.
            # return ValidationOutcome[OT].from_guard_history(call_log)
            return ValidationOutcome[OT](
                raw_llm_output=validation_output.raw_llm_response,  # type: ignore
                validated_output=cast(OT, validation_output.validated_output),
                validation_passed=validation_output.result,
            )
        else:
            raise ValueError("Guard does not have an api client!")
