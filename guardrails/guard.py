import contextvars
import json
import os
from builtins import id as object_id
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    overload,
)
from typing_extensions import deprecated
import warnings
from langchain_core.runnables import Runnable

from guardrails_api_client import (
    Guard as IGuard,
    ValidatorReference,
    ValidatePayload,
    SimpleTypes,
    ValidationOutcome as IValidationOutcome,
)
from opentelemetry import context as otel_context
from pydantic import field_validator
from pydantic.config import ConfigDict

from guardrails.api_client import GuardrailsApiClient
from guardrails.classes.output_type import OT
from guardrails.classes.rc import RC
from guardrails.classes.validation.validation_result import ErrorSpan
from guardrails.classes.validation.validation_summary import ValidationSummary
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.classes.execution import GuardExecutionOptions
from guardrails.classes.generic import Stack
from guardrails.classes.history import Call
from guardrails.classes.history.call_inputs import CallInputs
from guardrails.classes.output_type import OutputTypes
from guardrails.classes.schema.processed_schema import ProcessedSchema
from guardrails.classes.schema.model_schema import ModelSchema
from guardrails.formatters import BaseFormatter, get_formatter
from guardrails.llm_providers import (
    get_llm_api_enum,
    get_llm_ask,
    model_is_supported_server_side,
)
from guardrails.logger import logger, set_scope
from guardrails.run import Runner, StreamRunner
from guardrails.schema.primitive_schema import primitive_to_schema
from guardrails.schema.pydantic_schema import pydantic_model_to_schema
from guardrails.schema.rail_schema import rail_file_to_schema, rail_string_to_schema
from guardrails.schema.validator import SchemaValidationError, validate_json_schema
from guardrails.stores.context import (
    Tracer,
    Context,
    get_call_kwarg,
    get_tracer_context,
    set_call_kwargs,
    set_guard_name,
    set_tracer,
    set_tracer_context,
)
from guardrails.hub_telemetry.hub_tracing import trace
from guardrails.types.on_fail import OnFailAction
from guardrails.types.pydantic import ModelOrListOfModels
from guardrails.utils.naming_utils import random_id
from guardrails.utils.api_utils import extract_serializeable_metadata
from guardrails.utils.hub_telemetry_utils import HubTelemetry
from guardrails.telemetry import (
    trace_guard_execution,
    wrap_with_otel_context,
)
from guardrails.utils.validator_utils import (
    get_validator,
    parse_validator_reference,
    verify_metadata_requirements,
)
from guardrails.validator_base import Validator
from guardrails.types import (
    UseManyValidatorTuple,
    UseManyValidatorSpec,
    UseValidatorSpec,
    ValidatorMap,
)

from guardrails.utils.structured_data_utils import (
    # Prevent duplicate declaration in the docs
    json_function_calling_tool as json_function_calling_tool_util,
    output_format_json_schema as output_format_json_schema,
)

from guardrails.settings import settings
from guardrails.decorators.experimental import experimental


class Guard(IGuard, Generic[OT]):
    """The Guard class.

    This class is the main entry point for using Guardrails. It can be
    initialized by one of the following patterns:

    - `Guard().use(...)`
    - `Guard().use_many(...)`
    - `Guard.for_string(...)`
    - `Guard.for_pydantic(...)`
    - `Guard.for_rail(...)`
    - `Guard.for_rail_string(...)`

    The `__call__`
    method functions as a wrapper around LLM APIs. It takes in an LLM
    API, and optional prompt parameters, and returns a ValidationOutcome
    class that contains the raw output from
    the LLM, the validated output, as well as other helpful information.
    """

    validators: List[ValidatorReference]
    output_schema: ModelSchema
    history: Stack[Call]

    # Pydantic Config
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        validators: Optional[List[ValidatorReference]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Guard with serialized validator references and an
        output schema.

        Output schema must be a valid JSON Schema.
        """

        _try_to_load = name is not None

        # Shared Interface Properties
        id = id or random_id()
        name = name or f"gr-{id}"

        # Defaults
        validators = validators or []
        output_schema = output_schema or {"type": "string"}

        # Init ModelSchema class
        # schema_with_type = {**output_schema}
        # output_schema_type = output_schema.get("type")
        # if output_schema_type:
        #     schema_with_type["type"] = ValidationType.from_dict(output_schema_type)
        model_schema = ModelSchema.from_dict(output_schema)

        # TODO: Support a sink for history so that it is not solely held in memory
        history: Stack[Call] = Stack()

        # Super Init
        super().__init__(
            id=id,
            name=name,
            description=description,
            validators=validators,
            output_schema=model_schema,
            history=history,  # type: ignore - pyright doesn't understand pydantic overrides
        )

        ### Public ###
        ## Assigned in super ##
        # self.id: Optional[str] = None
        # self.name: Optional[str] = None
        # self.description: Optional[str] = None
        # self.validators: Optional[List[ValidatorReference]] = []
        # self.output_schema: Optional[ModelSchema] = None
        # self.history = history

        ### Legacy ##
        self._num_reasks = None
        self._rail: Optional[str] = None
        self._base_model: Optional[ModelOrListOfModels] = None

        ### Private ###
        self._validator_map: ValidatorMap = {}
        self._validators: List[Validator] = []
        self._output_type: OutputTypes = OutputTypes.__from_json_schema__(output_schema)
        self._exec_opts: GuardExecutionOptions = GuardExecutionOptions()
        self._tracer: Optional[Tracer] = None
        self._tracer_context: Optional[Context] = None
        self._hub_telemetry: HubTelemetry
        self._user_id: Optional[str] = None
        self._api_client: Optional[GuardrailsApiClient] = None
        self._allow_metrics_collection: Optional[bool] = None
        self._output_formatter: Optional[BaseFormatter] = None

        # Gaurdrails As A Service Initialization
        if settings.use_server:
            api_key = os.environ.get("GUARDRAILS_API_KEY")
            self._api_client = GuardrailsApiClient(api_key=api_key)
            _loaded = False
            if _try_to_load:
                loaded_guard = self._api_client.fetch_guard(self.name)
                if loaded_guard:
                    self.id = loaded_guard.id
                    self.description = loaded_guard.description
                    self.validators = loaded_guard.validators or []

                    loaded_output_schema = (
                        ModelSchema.from_dict(  # trims out extra keys
                            loaded_guard.output_schema.to_dict()
                            if loaded_guard.output_schema
                            else {"type": "string"}
                        )
                    )
                    self.output_schema = loaded_output_schema
                    _loaded = True
                else:
                    logger.warning(
                        f"use_server is True and Guard '{self.name}' "
                        "not found on the server. Creating a new empty Guard."
                    )
            if not _loaded:
                self._save()
        else:
            self.configure()

    @field_validator("output_schema")
    @classmethod
    def must_be_valid_json_schema(
        cls, output_schema: Optional[ModelSchema] = None
    ) -> Optional[ModelSchema]:
        if output_schema:
            try:
                validate_json_schema(output_schema.to_dict())
            except SchemaValidationError as e:
                raise ValueError(f"{str(e)}\n{json.dumps(e.fields, indent=2)}")
        return output_schema

    def configure(
        self,
        *,
        num_reasks: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        allow_metrics_collection: Optional[bool] = None,
    ):
        """Configure the Guard.

        Args:
            num_reasks (int, optional): The max times to re-ask the LLM
                if validation fails. Defaults to None.
            tracer (Tracer, optional): An OpenTelemetry tracer to use for
                sending traces to your OpenTelemetry sink. Defaults to None.
            allow_metrics_collection (bool, optional): Whether to allow
                Guardrails to collect anonymous metrics.
                Defaults to None, and falls back to waht is
                    set via the `guardrails configure` command.
        """
        if num_reasks:
            self._set_num_reasks(num_reasks)
        if tracer:
            self._set_tracer(tracer)
        self._load_rc()
        self._configure_hub_telemtry(allow_metrics_collection)

    def _set_num_reasks(self, num_reasks: Optional[int] = None) -> None:
        # Configure may check if num_reasks is none, but this method still needs to be
        # defensive for when it's called internally.  Setting a default parameter
        # doesn't help the case where the method is explicitly passed a 'None'.
        if num_reasks is None:
            logger.debug("_set_num_reasks called with 'None'.  Defaulting to 1.")
            self._num_reasks = 1
        else:
            self._num_reasks = num_reasks

    def _set_tracer(self, tracer: Optional[Tracer] = None) -> None:
        if tracer is not None:
            warnings.warn(
                "Setting tracer during initialization is deprecated"
                " and will be removed in 0.6.x!"
                "Configure a TracerProvider instead and we will"
                " obtain a tracer from it via the Tracer API.",
                DeprecationWarning,
            )
        self._tracer = tracer
        set_tracer(tracer)
        set_tracer_context()
        self._tracer_context = get_tracer_context()

    def _load_rc(self) -> None:
        rc = RC.load(logger)
        settings.rc = rc

    def _configure_hub_telemtry(
        self, allow_metrics_collection: Optional[bool] = None
    ) -> None:
        allow_metrics_collection = (
            settings.rc.enable_metrics is True
            if allow_metrics_collection is None
            else allow_metrics_collection
        )

        self._allow_metrics_collection = allow_metrics_collection

        # Initialize Hub Telemetry singleton and get the tracer
        self._hub_telemetry = HubTelemetry()
        self._hub_telemetry._enabled = allow_metrics_collection

        if allow_metrics_collection is True:
            # Get unique id of user from rc file
            self._user_id = settings.rc.id or ""

    def _fill_validator_map(self):
        # dont init validators if were going to call the server
        if settings.use_server:
            return
        for ref in self.validators:
            entry: List[Validator] = self._validator_map.get(ref.on, [])  # type: ignore
            # Check if the validator from the reference
            #   has an instance in the validator_map
            existing_instance: Optional[Validator] = None
            for v in entry:
                same_id = v.rail_alias == ref.id
                same_on_fail = v.on_fail_descriptor == ref.on_fail or (  # is default
                    v.on_fail_descriptor == OnFailAction.NOOP and not ref.on_fail
                )
                same_args = v.get_args() == ref.kwargs or (  # Both are empty
                    not v.get_args() and not ref.kwargs
                )
                if same_id and same_on_fail and same_args:
                    existing_instance = v
                    break
            if not existing_instance:
                validator = parse_validator_reference(ref)
                if validator:
                    entry.append(validator)
                self._validator_map[ref.on] = entry  # type: ignore

    def _fill_validators(self):
        self._validators = [
            v
            for v_list in [self._validator_map[k] for k in self._validator_map]
            for v in v_list
        ]

    def _fill_exec_opts(
        self,
        *,
        num_reasks: Optional[int] = None,
        messages: Optional[List[Dict]] = None,
        reask_messages: Optional[List[Dict]] = None,
        **kwargs,  # noqa
    ):
        """Backfill execution options from kwargs."""
        if num_reasks is not None:
            self._exec_opts.num_reasks = num_reasks
        if messages is not None:
            self._exec_opts.messages = messages
        if reask_messages is not None:
            self._exec_opts.reask_messages = reask_messages

    @classmethod
    def _for_rail_schema(
        cls,
        schema: ProcessedSchema,
        rail: str,
        *,
        num_reasks: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        guard = cls(
            name=name,
            description=description,
            output_schema=schema.json_schema,
            validators=schema.validators,
        )
        if schema.output_type == OutputTypes.STRING:
            guard = cast(Guard[str], guard)
        elif schema.output_type == OutputTypes.LIST:
            guard = cast(Guard[List], guard)
        else:
            guard = cast(Guard[Dict], guard)
        guard.configure(num_reasks=num_reasks, tracer=tracer)
        guard._validator_map = schema.validator_map
        guard._exec_opts = schema.exec_opts
        guard._output_type = schema.output_type
        guard._rail = rail
        guard._fill_validators()
        return guard

    @deprecated(
        "Use `for_rail` instead. This method will be removed in 0.6.x.", category=None
    )
    @classmethod
    def from_rail(cls, rail_file: str, *args, **kwargs):
        return cls.for_rail(rail_file, *args, **kwargs)

    @classmethod
    def for_rail(
        cls,
        rail_file: str,
        *,
        num_reasks: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Create a Guard using a `.rail` file to specify the output schema,
        prompt, etc.

        Args:
            rail_file: The path to the `.rail` file.
            num_reasks (int, optional): The max times to re-ask the LLM if validation fails. Deprecated
            tracer (Tracer, optional): An OpenTelemetry tracer to use for metrics and traces. Defaults to None.
            name (str, optional): A unique name for this Guard. Defaults to `gr-` + the object id.
            description (str, optional): A description for this Guard. Defaults to None.

        Returns:
            An instance of the `Guard` class.
        """  # noqa

        if num_reasks:
            warnings.warn(
                "Setting num_reasks during initialization is deprecated"
                " and will be removed in 0.6.x!"
                "We recommend setting num_reasks when calling guard()"
                " or guard.parse() instead."
                "If you insist on setting it at the Guard level,"
                " use 'Guard.configure()'.",
                DeprecationWarning,
            )

        # We have to set the tracer in the ContextStore before the Rail,
        #   and therefore the Validators, are initialized
        cls._set_tracer(cls, tracer)  # type: ignore

        schema = rail_file_to_schema(rail_file)
        return cls._for_rail_schema(
            schema,
            rail=rail_file,
            num_reasks=num_reasks,
            tracer=tracer,
            name=name,
            description=description,
        )

    @deprecated(
        "Use `for_rail_string` instead. This method will be removed in 0.6.x.",
        category=None,
    )
    @classmethod
    def from_rail_string(
        cls,
        rail_string: str,
        *args,
        **kwargs,
    ):
        return cls.for_rail_string(rail_string, *args, **kwargs)

    @classmethod
    def for_rail_string(
        cls,
        rail_string: str,
        *,
        num_reasks: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Create a Guard using a `.rail` string to specify the output schema,
        prompt, etc..

        Args:
            rail_string: The `.rail` string.
            num_reasks (int, optional): The max times to re-ask the LLM if validation fails. Deprecated
            tracer (Tracer, optional): An OpenTelemetry tracer to use for metrics and traces. Defaults to None.
            name (str, optional): A unique name for this Guard. Defaults to `gr-` + the object id.
            description (str, optional): A description for this Guard. Defaults to None.

        Returns:
            An instance of the `Guard` class.
        """  # noqa

        if num_reasks:
            warnings.warn(
                "Setting num_reasks during initialization is deprecated"
                " and will be removed in 0.6.x!"
                "We recommend setting num_reasks when calling guard()"
                " or guard.parse() instead."
                "If you insist on setting it at the Guard level,"
                " use 'Guard.configure()'.",
                DeprecationWarning,
            )

        # We have to set the tracer in the ContextStore before the Rail,
        #   and therefore the Validators, are initialized
        cls._set_tracer(cls, tracer)  # type: ignore

        schema = rail_string_to_schema(rail_string)
        return cls._for_rail_schema(
            schema,
            rail=rail_string,
            num_reasks=num_reasks,
            tracer=tracer,
            name=name,
            description=description,
        )

    @deprecated(
        "Use `for_pydantic` instead. This method will be removed in 0.6.x.",
        category=None,
    )
    @classmethod
    def from_pydantic(cls, output_class: ModelOrListOfModels, *args, **kwargs):
        return cls.for_pydantic(output_class, **kwargs)

    @classmethod
    def for_pydantic(
        cls,
        output_class: ModelOrListOfModels,
        *,
        num_reasks: Optional[int] = None,
        reask_messages: Optional[List[Dict]] = None,
        messages: Optional[List[Dict]] = None,
        tracer: Optional[Tracer] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        output_formatter: Optional[Union[str, BaseFormatter]] = None,
    ):
        """Create a Guard instance using a Pydantic model to specify the output
        schema.

        Args:
            output_class: (Union[Type[BaseModel], List[Type[BaseModel]]]): The pydantic model that describes
            the desired structure of the output.
            messages (List[Dict], optional): A list of messages to give to the llm. Defaults to None.
            reask_messages (List[Dict], optional): A list of messages to use during reasks. Defaults to None.
            num_reasks (int, optional): The max times to re-ask the LLM if validation fails. Deprecated
            tracer (Tracer, optional): An OpenTelemetry tracer to use for metrics and traces. Defaults to None.
            name (str, optional): A unique name for this Guard. Defaults to `gr-` + the object id.
            description (str, optional): A description for this Guard. Defaults to None.
            output_formatter (str | Formatter, optional): 'none' (default), 'jsonformer', or a Guardrails Formatter.
        """  # noqa

        if num_reasks:
            warnings.warn(
                "Setting num_reasks during initialization is deprecated"
                " and will be removed in 0.6.x!"
                "We recommend setting num_reasks when calling guard()"
                " or guard.parse() instead."
                "If you insist on setting it at the Guard level,"
                " use 'Guard.configure()'.",
                DeprecationWarning,
            )

        # We have to set the tracer in the ContextStore before the Rail,
        #   and therefore the Validators, are initialized
        cls._set_tracer(cls, tracer)  # type: ignore

        schema = pydantic_model_to_schema(output_class)
        exec_opts = GuardExecutionOptions(
            reask_messages=reask_messages,
            messages=messages,
        )
        guard = cls(
            name=name,
            description=description,
            output_schema=schema.json_schema,
            validators=schema.validators,
        )
        if schema.output_type == OutputTypes.LIST:
            guard = cast(Guard[List], guard)
        else:
            guard = cast(Guard[Dict], guard)
        guard.configure(num_reasks=num_reasks, tracer=tracer)
        guard._validator_map = schema.validator_map
        guard._exec_opts = exec_opts
        guard._output_type = schema.output_type
        guard._base_model = output_class
        if isinstance(output_formatter, str):
            if isinstance(output_class, list):
                raise Exception("""Root-level arrays are not supported with the 
                jsonformer argument, but can be used with other json generation methods.
                Omit the output_formatter argument to use the other methods.""")
            output_formatter = get_formatter(
                output_formatter,
                schema=output_class.model_json_schema(),  # type: ignore
            )
        guard._output_formatter = output_formatter
        guard._fill_validators()
        return guard

    @deprecated(
        """Use `use`, `use_many`, or `for_string` instead.
                This method will be removed in 0.6.x.""",
        category=None,
    )
    @classmethod
    def from_string(
        cls,
        validators: Sequence[Validator],
        *args,
        **kwargs,
    ):
        return cls.for_string(validators, *args, **kwargs)

    @classmethod
    def for_string(
        cls,
        validators: Sequence[Validator],
        *,
        string_description: Optional[str] = None,
        reask_messages: Optional[List[Dict]] = None,
        messages: Optional[List[Dict]] = None,
        num_reasks: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Create a Guard instance for a string response.

        Args:
            validators: (List[Validator]): The list of validators to apply to the string output.
            string_description (str, optional): A description for the string to be generated. Defaults to None.
            messages (List[Dict], optional): A list of messages to pass to llm. Defaults to None.
            reask_messages (List[Dict], optional): A list of messages to use during reasks. Defaults to None.
            num_reasks (int, optional): The max times to re-ask the LLM if validation fails. Deprecated
            tracer (Tracer, optional): An OpenTelemetry tracer to use for metrics and traces. Defaults to None.
            name (str, optional): A unique name for this Guard. Defaults to `gr-` + the object id.
            description (str, optional): A description for this Guard. Defaults to None.
        """  # noqa
        if num_reasks:
            warnings.warn(
                "Setting num_reasks during initialization is deprecated"
                " and will be removed in 0.6.x!"
                "We recommend setting num_reasks when calling guard()"
                " or guard.parse() instead."
                "If you insist on setting it at the Guard level,"
                " use 'Guard.configure()'.",
                DeprecationWarning,
            )

        # This might not be necessary anymore
        cls._set_tracer(cls, tracer)  # type: ignore

        schema = primitive_to_schema(
            list(validators), type=SimpleTypes.STRING, description=string_description
        )
        exec_opts = GuardExecutionOptions(
            messages=messages,
            reask_messages=reask_messages,
        )
        guard = cast(
            Guard[str],
            cls(
                name=name,
                description=description,
                output_schema=schema.json_schema,
                validators=schema.validators,
            ),
        )
        guard.configure(num_reasks=num_reasks, tracer=tracer)
        guard._validator_map = schema.validator_map
        guard._exec_opts = exec_opts
        guard._output_type = schema.output_type
        guard._fill_validators()
        return guard

    def _execute(
        self,
        *args,
        llm_api: Optional[Callable] = None,
        llm_output: Optional[str] = None,
        prompt_params: Optional[Dict] = None,
        num_reasks: Optional[int] = None,
        messages: Optional[List[Dict]] = None,
        reask_messages: Optional[List[Dict]] = None,
        metadata: Optional[Dict],
        full_schema_reask: Optional[bool] = None,
        **kwargs,
    ) -> Union[ValidationOutcome[OT], Iterator[ValidationOutcome[OT]]]:
        self._fill_validator_map()
        self._fill_validators()
        self._fill_exec_opts(
            num_reasks=num_reasks,
            messages=messages,
            reask_messages=reask_messages,
        )
        metadata = metadata or {}
        # if not llm_output and llm_api and not (messages):
        #     raise RuntimeError("'messages' must be provided in order to call an LLM!")

        # check if validator requirements are fulfilled
        missing_keys = verify_metadata_requirements(metadata, self._validators)
        if missing_keys:
            raise ValueError(
                f"Missing required metadata keys: {', '.join(missing_keys)}"
            )

        def __exec(
            self: Guard,
            *args,
            llm_api: Optional[Callable] = None,
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

            set_call_kwargs(kwargs)
            set_tracer(self._tracer)
            set_tracer_context(self._tracer_context)
            set_guard_name(self.name)

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

            if settings.use_server and model_is_supported_server_side(
                llm_api, *args, **kwargs
            ):
                return self._call_server(
                    llm_output=llm_output,
                    llm_api=llm_api,
                    num_reasks=self._num_reasks,
                    prompt_params=prompt_params,
                    metadata=metadata,
                    full_schema_reask=full_schema_reask,
                    *args,
                    **kwargs,
                )

            call_log = Call(inputs=call_inputs)
            set_scope(str(object_id(call_log)))
            self.history.push(call_log)
            # Otherwise, call the LLM synchronously
            return self._exec(
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

        guard_context = contextvars.Context()

        # get the current otel context and wrap the subsequent call
        #   to preserve otel context if guard call is being called be another
        # framework upstream
        current_otel_context = otel_context.get_current()
        wrapped__exec = wrap_with_otel_context(current_otel_context, __exec)

        return guard_context.run(
            wrapped__exec,
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

    def _exec(
        self,
        *args,
        llm_api: Optional[Callable] = None,
        llm_output: Optional[str] = None,
        call_log: Call,  # Not optional, but internal
        prompt_params: Dict,  # Should be defined at this point
        num_reasks: int = 0,  # Should be defined at this point
        metadata: Dict,  # Should be defined at this point
        full_schema_reask: bool = False,  # Should be defined at this point
        messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Union[ValidationOutcome[OT], Iterator[ValidationOutcome[OT]]]:
        api = None

        if llm_api is not None or kwargs.get("model") is not None:
            api = get_llm_ask(llm_api, *args, **kwargs)

        if self._output_formatter is not None:
            # Type suppression here? ArbitraryCallable is a subclass of PromptCallable!?
            api = self._output_formatter.wrap_callable(api)  # type: ignore

        # Check whether stream is set
        if kwargs.get("stream", False):
            # If stream is True, use StreamRunner
            runner = StreamRunner(
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
                disable_tracer=(
                    not self._allow_metrics_collection
                    if isinstance(self._allow_metrics_collection, bool)
                    else None
                ),
                exec_options=self._exec_opts,
            )
            return runner(call_log=call_log, prompt_params=prompt_params)
        else:
            # Otherwise, use Runner
            runner = Runner(
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
                disable_tracer=(
                    not self._allow_metrics_collection
                    if isinstance(self._allow_metrics_collection, bool)
                    else None
                ),
                exec_options=self._exec_opts,
            )
            call = runner(call_log=call_log, prompt_params=prompt_params)
            return ValidationOutcome[OT].from_guard_history(call)

    @trace(name="/guard_call", origin="Guard.__call__")
    def __call__(
        self,
        llm_api: Optional[Callable] = None,
        *args,
        prompt_params: Optional[Dict] = None,
        num_reasks: Optional[int] = 1,
        messages: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
        **kwargs,
    ) -> Union[ValidationOutcome[OT], Iterator[ValidationOutcome[OT]]]:
        """Call the LLM and validate the output.

        Args:
            llm_api: The LLM API to call
                     (e.g. openai.completions.create or openai.Completion.acreate)
            prompt_params: The parameters to pass to the prompt.format() method.
            num_reasks: The max times to re-ask the LLM for invalid output.
            messages: The message history to pass to the LLM.
            metadata: Metadata to pass to the validators.
            full_schema_reask: When reasking, whether to regenerate the full schema
                               or just the incorrect values.
                               Defaults to `True` if a base model is provided,
                               `False` otherwise.

        Returns:
            ValidationOutcome
        """

        messages = messages or self._exec_opts.messages or []

        if messages is not None and not len(messages):
            raise RuntimeError(
                "You must provide messages. "
                "Alternatively, you can provide messages in the Schema constructor."
            )

        return trace_guard_execution(
            self.name,
            self.history,
            self._execute,
            self._tracer,
            *args,
            llm_api=llm_api,
            prompt_params=prompt_params,
            num_reasks=num_reasks,
            messages=messages,
            metadata=metadata,
            full_schema_reask=full_schema_reask,
            **kwargs,
        )

    @trace(name="/guard_call", origin="Guard.parse")
    def parse(
        self,
        llm_output: str,
        *args,
        metadata: Optional[Dict] = None,
        llm_api: Optional[Callable] = None,
        num_reasks: Optional[int] = None,
        prompt_params: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
        **kwargs,
    ) -> ValidationOutcome[OT]:
        """Alternate flow to using Guard where the llm_output is known.

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
            ValidationOutcome
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

        return trace_guard_execution(
            self.name,
            self.history,
            self._execute,  # type: ignore # streams are supported for parse
            self._tracer,
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

    def error_spans_in_output(self) -> List[ErrorSpan]:
        """Get the error spans in the last output."""
        try:
            call = self.history.last
            if call:
                iter = call.iterations.last
                if iter:
                    llm_spans = iter.error_spans_in_output
                    return llm_spans
            return []
        except (AttributeError, TypeError):
            return []

    def __add_validator(self, validator: Validator, on: str = "output"):
        if on not in [
            "output",
            "messages",
        ] and not on.startswith("$"):
            warnings.warn(
                f"Unusual 'on' value: {on}!"
                "This value is typically one of "
                "'output', 'messages') "
                "or a JSON path starting with '$.'",
                UserWarning,
            )

        if on == "output":
            on = "$"

        validator_reference = ValidatorReference(
            id=validator.rail_alias,
            on=on,
            on_fail=validator.on_fail_descriptor,  # type: ignore
            kwargs=validator.get_args(),
        )
        self.validators.append(validator_reference)
        self._validator_map[on] = self._validator_map.get(on, [])
        self._validator_map[on].append(validator)
        self._validators.append(validator)

    @overload
    def use(self, validator: Validator, *, on: str = "output") -> "Guard": ...

    @overload
    def use(
        self, validator: Type[Validator], *args, on: str = "output", **kwargs
    ) -> "Guard": ...

    def use(
        self,
        validator: UseValidatorSpec,
        *args,
        on: str = "output",
        **kwargs,
    ) -> "Guard":
        """Use a validator to validate either of the following:
        - The output of an LLM request
        - The message history

        Args:
            validator: The validator to use. Either the class or an instance.
            on: The part of the LLM request to validate. Defaults to "output".
        """
        # check if args has any validators hiding in it
        # throw error to user so they can update
        if args:
            for arg in args:
                if (
                    isinstance(arg, type)
                    and issubclass(arg, Validator)
                    or isinstance(arg, Validator)
                ):
                    raise ValueError(
                        "Validator is an argument besides the first."
                        "Please pass it as the first or use the 'use_many' method for"
                        " multiple validators."
                    )

        hydrated_validator = get_validator(validator, *args, **kwargs)
        self.__add_validator(hydrated_validator, on=on)
        self._save()
        return self

    @overload
    def use_many(self, *validators: Validator, on: str = "output") -> "Guard": ...

    @overload
    def use_many(
        self,
        *validators: UseManyValidatorTuple,
        on: str = "output",
    ) -> "Guard": ...

    def use_many(
        self,
        *validators: UseManyValidatorSpec,
        on: str = "output",
    ) -> "Guard":
        """Use multiple validators to validate results of an LLM request."""
        # Loop through the validators
        for v in validators:
            hydrated_validator = get_validator(v)
            self.__add_validator(hydrated_validator, on=on)
        self._save()
        return self

    @trace(name="/guard_call", origin="Guard.validate")
    def validate(self, llm_output: str, *args, **kwargs) -> ValidationOutcome[OT]:
        return self.parse(llm_output=llm_output, *args, **kwargs)

    # No call support for this until
    # https://github.com/guardrails-ai/guardrails/pull/525 is merged
    # def __call__(self, llm_output: str, *args, **kwargs) -> ValidationOutcome[str]:
    #     return self.validate(llm_output, *args, **kwargs)

    # TODO: Test generated history and override to_dict if necessary
    # def to_dict(self) -> Dict[str, Any]:
    #     pass

    def upsert_guard(self):
        if settings.use_server and self._api_client:
            self._api_client.upsert_guard(self)
        else:
            raise ValueError("Using the Guardrails server is not enabled!")

    def _single_server_call(self, *, payload: Dict[str, Any]) -> ValidationOutcome[OT]:
        if settings.use_server and self._api_client:
            validation_output: IValidationOutcome = self._api_client.validate(
                guard=self,  # type: ignore
                payload=ValidatePayload.from_dict(payload),  # type: ignore
                openai_api_key=get_call_kwarg("api_key"),
            )
            if not validation_output:
                return ValidationOutcome[OT](
                    call_id="0",  # type: ignore
                    raw_llm_output=None,
                    validated_output=None,
                    validation_passed=False,
                    error="The response from the server was empty!",
                )
            if os.environ.get("GUARD_HISTORY_ENABLED", "true").lower() == "true":
                guard_history = self._api_client.get_history(
                    self.name, validation_output.call_id
                )
                self.history.extend(
                    [Call.from_interface(call) for call in guard_history]
                )

            validation_summaries = []
            if self.history.last and self.history.last.iterations.last:
                validator_logs = self.history.last.iterations.last.validator_logs
                validation_summaries = ValidationSummary.from_validator_logs_only_fails(
                    validator_logs
                )

            # TODO: See if the below statement is still true
            # Our interfaces are too different for this to work right now.
            # Once we move towards shared interfaces for both the open source
            # and the api we can re-enable this.
            # return ValidationOutcome[OT].from_guard_history(call_log)
            validated_output = (
                cast(OT, validation_output.validated_output.actual_instance)
                if validation_output.validated_output
                else None
            )
            return ValidationOutcome[OT](
                call_id=validation_output.call_id,  # type: ignore
                raw_llm_output=validation_output.raw_llm_output,
                validated_output=validated_output,
                validation_passed=(validation_output.validation_passed is True),
                validation_summaries=validation_summaries,
            )
        else:
            raise ValueError("Guard does not have an api client!")

    def _stream_server_call(
        self,
        *,
        payload: Dict[str, Any],
    ) -> Iterator[ValidationOutcome[OT]]:
        if settings.use_server and self._api_client:
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
                        raw_llm_output=validation_output.raw_llm_output,
                        validated_output=validated_output,
                        validation_passed=(validation_output.validation_passed is True),
                    )

            if os.environ.get("GUARD_HISTORY_ENABLED", "true").lower() == "true":
                if validation_output:
                    guard_history = self._api_client.get_history(
                        self.name, validation_output.call_id
                    )
                    self.history.extend(
                        [Call.from_interface(call) for call in guard_history]
                    )
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
        **kwargs,
    ) -> Union[ValidationOutcome[OT], Iterator[ValidationOutcome[OT]]]:
        if settings.use_server and self._api_client:
            payload: Dict[str, Any] = {
                "args": list(args),
                "full_schema_reask": full_schema_reask,
            }
            payload.update(**kwargs)
            if metadata:
                payload["metadata"] = extract_serializeable_metadata(metadata)
            if llm_output is not None:
                payload["llmOutput"] = llm_output
            if num_reasks is not None:
                payload["numReasks"] = num_reasks or self._exec_opts.num_reasks
            if prompt_params is not None:
                payload["promptParams"] = prompt_params
            if llm_api is not None:
                payload["llmApi"] = get_llm_api_enum(llm_api, *args, **kwargs)

            if not payload.get("messages"):
                payload["messages"] = self._exec_opts.messages
            if not payload.get("reask_messages"):
                payload["reask_messages"] = self._exec_opts.reask_messages

            should_stream = kwargs.get("stream", False)
            if should_stream:
                return self._stream_server_call(payload=payload)
            else:
                return self._single_server_call(
                    payload=payload,
                )
        else:
            raise ValueError("Guard does not have an api client!")

    def _save(self):
        api_key = os.environ.get("GUARDRAILS_API_KEY")
        if settings.use_server:
            if self.name is None:
                self.name = f"gr-{str(self.id)}"
                logger.warning("No name passed to guard!")
                logger.warning(
                    "Use this auto-generated name to re-use this guard: {name}".format(
                        name=self.name
                    )
                )
            if not self._api_client:
                self._api_client = GuardrailsApiClient(api_key=api_key)
            self.upsert_guard()

    def to_runnable(self) -> Runnable:
        """Convert a Guard to a LangChain Runnable."""
        from guardrails.integrations.langchain.guard_runnable import GuardRunnable

        return GuardRunnable(self)

    # override IGuard.to_dict
    def to_dict(self) -> Dict[str, Any]:
        i_guard = IGuard(
            id=self.id,
            name=self.name,
            description=self.description,
            validators=self.validators,
            output_schema=self.output_schema,
            history=[c.to_interface() for c in self.history],  # type: ignore
        )

        return i_guard.to_dict()

    @experimental
    def response_format_json_schema(self) -> Dict[str, Any]:
        return output_format_json_schema(schema=self._base_model)  # type: ignore

    def json_function_calling_tool(
        self,
        tools: Optional[list] = None,
    ) -> List[Dict[str, Any]]:
        """Appends an OpenAI tool that specifies the output structure using
        JSON Schema for chat models."""
        tools = json_function_calling_tool_util(
            tools=tools,
            # todo to_dict has a slight bug workaround here
            # but should fix in the long run dont have to
            # serialize and deserialize
            schema=json.loads(self.output_schema.to_json()),
        )
        return tools

    # override IGuard.from_dict
    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional["Guard"]:
        i_guard = IGuard.from_dict(obj)
        if not i_guard:
            return i_guard
        output_schema = (
            i_guard.output_schema.to_dict() if i_guard.output_schema else None
        )

        guard = cls(
            id=i_guard.id,
            name=i_guard.name,
            description=i_guard.description,
            validators=i_guard.validators,
            output_schema=output_schema,
        )

        history = (
            [Call.from_interface(i_call) for i_call in i_guard.history]
            if i_guard.history
            else []
        )
        guard.history = Stack(*history)
        return guard

    # attempts to get a guard from the server
    # if a name is unspecified, the guard will be created on the client
    # in the future, this may create a guard on the server
    @experimental
    @staticmethod
    def fetch_guard(
        name: Optional[str] = None,
        *args,
        **kwargs,
    ):
        if not name:
            raise ValueError("Name must be specified to fetch a guard")

        settings.use_server = True
        api_key = os.environ.get("GUARDRAILS_API_KEY")
        api_client = GuardrailsApiClient(api_key=api_key)
        guard = api_client.fetch_guard(name)
        if guard:
            return Guard(name=name, *args, **kwargs)

        raise ValueError(f"Guard with name {name} not found")
