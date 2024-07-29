import contextvars
import json
import os
from builtins import id as object_id
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    overload,
)
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
from guardrails.classes.validation.validation_result import ErrorSpan
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.classes.credentials import Credentials
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
    set_tracer,
    set_tracer_context,
)
from guardrails.types.on_fail import OnFailAction
from guardrails.types.pydantic import ModelOrListOfModels
from guardrails.utils.naming_utils import random_id
from guardrails.utils.api_utils import extract_serializeable_metadata
from guardrails.utils.hub_telemetry_utils import HubTelemetry
from guardrails.utils.telemetry_utils import wrap_with_otel_context
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

from guardrails.utils.tools_utils import (
    # Prevent duplicate declaration in the docs
    json_function_calling_tool as json_function_calling_tool_util,
)
from guardrails.settings import settings


class Guard(IGuard, Generic[OT]):
    """The Guard class.

    This class is the main entry point for using Guardrails. It can be
    initialized by one of the following patterns:

    - `Guard().use(...)`
    - `Guard().use_many(...)`
    - `Guard.from_string(...)`
    - `Guard.from_pydantic(...)`
    - `Guard.from_rail(...)`
    - `Guard.from_rail_string(...)`

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
            if not _loaded:
                self._save()

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
        """Configure the Guard."""
        if num_reasks:
            self._set_num_reasks(num_reasks)
        if tracer:
            self._set_tracer(tracer)
        self._configure_telemtry(allow_metrics_collection)

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
        self._tracer = tracer
        set_tracer(tracer)
        set_tracer_context()
        self._tracer_context = get_tracer_context()

    def _configure_telemtry(
        self, allow_metrics_collection: Optional[bool] = None
    ) -> None:
        credentials = None
        if allow_metrics_collection is None:
            credentials = Credentials.from_rc_file(logger)
            # TODO: Check credentials.enable_metrics after merge from main
            allow_metrics_collection = credentials.enable_metrics is True

        self._allow_metrics_collection = allow_metrics_collection

        if allow_metrics_collection:
            if not credentials:
                credentials = Credentials.from_rc_file(logger)
            # Get unique id of user from credentials
            self._user_id = credentials.id or ""
            # Initialize Hub Telemetry singleton and get the tracer
            self._hub_telemetry = HubTelemetry()

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
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        reask_prompt: Optional[str] = None,
        reask_instructions: Optional[str] = None,
        **kwargs,  # noqa
    ):
        """Backfill execution options from kwargs."""
        if num_reasks is not None:
            self._exec_opts.num_reasks = num_reasks
        if prompt is not None:
            self._exec_opts.prompt = prompt
        if instructions is not None:
            self._exec_opts.instructions = instructions
        if msg_history is not None:
            self._exec_opts.msg_history = msg_history
        if reask_prompt is not None:
            self._exec_opts.reask_prompt = reask_prompt
        if reask_instructions is not None:
            self._exec_opts.reask_instructions = reask_instructions

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

    @classmethod
    def from_rail(
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
        return cls._from_rail_schema(
            schema,
            rail=rail_file,
            num_reasks=num_reasks,
            tracer=tracer,
            name=name,
            description=description,
        )

    @classmethod
    def from_rail_string(
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
        return cls._from_rail_schema(
            schema,
            rail=rail_string,
            num_reasks=num_reasks,
            tracer=tracer,
            name=name,
            description=description,
        )

    @classmethod
    def from_pydantic(
        cls,
        output_class: ModelOrListOfModels,
        *,
        prompt: Optional[str] = None,  # TODO: deprecate this in 0.5.1
        instructions: Optional[str] = None,  # TODO: deprecate this in 0.5.1
        num_reasks: Optional[int] = None,
        reask_prompt: Optional[str] = None,  # TODO: deprecate this in 0.5.1
        reask_instructions: Optional[str] = None,  # TODO: deprecate this in 0.5.1
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
            prompt (str, optional): The prompt used to generate the string. Defaults to None.
            instructions (str, optional): Instructions for chat models. Defaults to None.
            reask_prompt (str, optional): An alternative prompt to use during reasks. Defaults to None.
            reask_instructions (str, optional): Alternative instructions to use during reasks. Defaults to None.
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
            prompt=prompt,
            instructions=instructions,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
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

    @classmethod
    def from_string(
        cls,
        validators: Sequence[Validator],
        *,
        string_description: Optional[str] = None,
        prompt: Optional[str] = None,  # TODO: deprecate this in 0.5.1
        instructions: Optional[str] = None,  # TODO: deprecate this in 0.5.1
        reask_prompt: Optional[str] = None,  # TODO: deprecate this in 0.5.1
        reask_instructions: Optional[str] = None,  # TODO: deprecate this in 0.5.1
        num_reasks: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Create a Guard instance for a string response.

        Args:
            validators: (List[Validator]): The list of validators to apply to the string output.
            string_description (str, optional): A description for the string to be generated. Defaults to None.
            prompt (str, optional): The prompt used to generate the string. Defaults to None.
            instructions (str, optional): Instructions for chat models. Defaults to None.
            reask_prompt (str, optional): An alternative prompt to use during reasks. Defaults to None.
            reask_instructions (str, optional): Alternative instructions to use during reasks. Defaults to None.
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
            prompt=prompt,
            instructions=instructions,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
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
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        reask_prompt: Optional[str] = None,
        reask_instructions: Optional[str] = None,
        metadata: Optional[Dict],
        full_schema_reask: Optional[bool] = None,
        **kwargs,
    ) -> Union[ValidationOutcome[OT], Iterable[ValidationOutcome[OT]]]:
        self._fill_validator_map()
        self._fill_validators()
        self._fill_exec_opts(
            num_reasks=num_reasks,
            prompt=prompt,
            instructions=instructions,
            msg_history=msg_history,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
        )
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

        def __exec(
            self: Guard,
            *args,
            llm_api: Optional[Callable] = None,
            llm_output: Optional[str] = None,
            prompt_params: Optional[Dict] = None,
            num_reasks: Optional[int] = None,
            prompt: Optional[str] = None,
            instructions: Optional[str] = None,
            msg_history: Optional[List[Dict]] = None,
            metadata: Optional[Dict] = None,
            full_schema_reask: Optional[bool] = None,
            **kwargs,
        ):
            prompt_params = prompt_params or {}
            metadata = metadata or {}
            if full_schema_reask is None:
                full_schema_reask = self._base_model is not None

            if self._allow_metrics_collection and self._hub_telemetry:
                # Create a new span for this guard call
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
                self._hub_telemetry.create_new_span(
                    span_name="/guard_call",
                    attributes=[
                        ("guard_id", self.id),
                        ("user_id", self._user_id),
                        ("llm_api", llm_api_str if llm_api_str else "None"),
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
                    prompt=prompt,
                    instructions=instructions,
                    msg_history=msg_history,
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
            prompt=prompt,
            instructions=instructions,
            msg_history=msg_history,
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
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Union[ValidationOutcome[OT], Iterable[ValidationOutcome[OT]]]:
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
            return runner(call_log=call_log, prompt_params=prompt_params)
        else:
            # Otherwise, use Runner
            runner = Runner(
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
            call = runner(call_log=call_log, prompt_params=prompt_params)
            return ValidationOutcome[OT].from_guard_history(call)

    def __call__(
        self,
        llm_api: Optional[Callable] = None,
        *args,
        prompt_params: Optional[Dict] = None,
        num_reasks: Optional[int] = 1,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
        **kwargs,
    ) -> Union[ValidationOutcome[OT], Iterable[ValidationOutcome[OT]]]:
        """Call the LLM and validate the output.

        Args:
            llm_api: The LLM API to call
                     (e.g. openai.completions.create or openai.Completion.acreate)
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
            ValidationOutcome
        """
        instructions = instructions or self._exec_opts.instructions
        prompt = prompt or self._exec_opts.prompt
        msg_history = msg_history or kwargs.get("messages", None) or []
        if prompt is None:
            if msg_history is not None and not len(msg_history):
                raise RuntimeError(
                    "You must provide a prompt if msg_history is empty. "
                    "Alternatively, you can provide a prompt in the Schema constructor."
                )

        return self._execute(
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
        default_prompt = self._exec_opts.prompt if llm_api else None
        prompt = kwargs.pop("prompt", default_prompt)

        default_instructions = self._exec_opts.instructions if llm_api else None
        instructions = kwargs.pop("instructions", default_instructions)

        default_msg_history = self._exec_opts.msg_history if llm_api else None
        msg_history = kwargs.pop("msg_history", default_msg_history)

        return self._execute(  # type: ignore # streams are supported for parse
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
            "prompt",
            "instructions",
            "msg_history",
        ] and not on.startswith("$"):
            warnings.warn(
                f"Unusual 'on' value: {on}!"
                "This value is typically one of "
                "'output', 'prompt', 'instructions', 'msg_history') "
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
        - The prompt
        - The instructions
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
        if on == "messages":
            on = "msg_history"
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
        if on == "messages":
            on = "msg_history"
        for v in validators:
            hydrated_validator = get_validator(v)
            self.__add_validator(hydrated_validator, on=on)
        self._save()
        return self

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

            guard_history = self._api_client.get_history(
                self.name, validation_output.call_id
            )
            self.history.extend([Call.from_interface(call) for call in guard_history])

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
            )
        else:
            raise ValueError("Guard does not have an api client!")

    def _stream_server_call(
        self,
        *,
        payload: Dict[str, Any],
    ) -> Iterable[ValidationOutcome[OT]]:
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
    ) -> Union[ValidationOutcome[OT], Iterable[ValidationOutcome[OT]]]:
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

            if not payload.get("prompt"):
                payload["prompt"] = self._exec_opts.prompt
            if not payload.get("instructions"):
                payload["instructions"] = self._exec_opts.instructions
            if not payload.get("msg_history"):
                payload["msg_history"] = self._exec_opts.msg_history
            if not payload.get("reask_prompt"):
                payload["reask_prompt"] = self._exec_opts.reask_prompt
            if not payload.get("reask_instructions"):
                payload["reask_instructions"] = self._exec_opts.reask_instructions

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
