import asyncio
import contextvars
import warnings
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
    Type,
    Union,
    cast,
    overload,
)

from eliot import add_destinations, start_action
from pydantic import BaseModel

from guardrails.classes import OT, ValidationOutcome
from guardrails.classes.generic import Stack
from guardrails.classes.history import Call
from guardrails.classes.history.call_inputs import CallInputs
from guardrails.llm_providers import get_async_llm_ask, get_llm_ask
from guardrails.logger import logger, set_scope
from guardrails.prompt import Instructions, Prompt
from guardrails.rail import Rail
from guardrails.run import AsyncRunner, Runner, StreamRunner
from guardrails.schema import Schema, StringSchema
from guardrails.validators import Validator

add_destinations(logger.debug)


class Guard(Generic[OT]):
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

    def __init__(
        self,
        rail: Rail,
        num_reasks: Optional[int] = None,
        base_model: Optional[Type[BaseModel]] = None,
    ):
        """Initialize the Guard."""
        self.rail = rail
        self.num_reasks = num_reasks
        # TODO: Support a sink for history so that it is not solely held in memory
        self.history: Stack[Call] = Stack()
        self.base_model = base_model

    @property
    def prompt_schema(self) -> Optional[StringSchema]:
        """Return the input schema."""
        return self.rail.prompt_schema

    @property
    def instructions_schema(self) -> Optional[StringSchema]:
        """Return the input schema."""
        return self.rail.instructions_schema

    @property
    def msg_history_schema(self) -> Optional[StringSchema]:
        """Return the input schema."""
        return self.rail.msg_history_schema

    @property
    def output_schema(self) -> Schema:
        """Return the output schema."""
        return self.rail.output_schema

    @property
    def instructions(self) -> Optional[Instructions]:
        """Return the instruction-prompt."""
        return self.rail.instructions

    @property
    def prompt(self) -> Optional[Prompt]:
        """Return the prompt."""
        return self.rail.prompt

    @property
    def raw_prompt(self) -> Optional[Prompt]:
        """Return the prompt, alias for `prompt`."""
        return self.prompt

    @property
    def base_prompt(self) -> Optional[str]:
        """Return the base prompt i.e. prompt.source."""
        if self.prompt is None:
            return None
        return self.prompt.source

    @property
    def reask_prompt(self) -> Optional[Prompt]:
        """Return the reask prompt."""
        return self.output_schema.reask_prompt_template

    @reask_prompt.setter
    def reask_prompt(self, reask_prompt: Optional[str]):
        """Set the reask prompt."""
        self.output_schema.reask_prompt_template = reask_prompt

    @property
    def reask_instructions(self) -> Optional[Instructions]:
        """Return the reask prompt."""
        return self.output_schema.reask_instructions_template

    @reask_instructions.setter
    def reask_instructions(self, reask_instructions: Optional[str]):
        """Set the reask prompt."""
        self.output_schema.reask_instructions_template = reask_instructions

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

    @classmethod
    def from_rail(cls, rail_file: str, num_reasks: Optional[int] = None):
        """Create a Schema from a `.rail` file.

        Args:
            rail_file: The path to the `.rail` file.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            An instance of the `Guard` class.
        """
        rail = Rail.from_file(rail_file)
        if rail.output_type == "str":
            return cast(Guard[str], cls(rail=rail, num_reasks=num_reasks))
        return cast(Guard[Dict], cls(rail=rail, num_reasks=num_reasks))

    @classmethod
    def from_rail_string(cls, rail_string: str, num_reasks: Optional[int] = None):
        """Create a Schema from a `.rail` string.

        Args:
            rail_string: The `.rail` string.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            An instance of the `Guard` class.
        """
        rail = Rail.from_string(rail_string)
        if rail.output_type == "str":
            return cast(Guard[str], cls(rail=rail, num_reasks=num_reasks))
        return cast(Guard[Dict], cls(rail=rail, num_reasks=num_reasks))

    @classmethod
    def from_pydantic(
        cls,
        output_class: Type[BaseModel],
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        num_reasks: Optional[int] = None,
        reask_prompt: Optional[str] = None,
        reask_instructions: Optional[str] = None,
    ):
        """Create a Guard instance from a Pydantic model and prompt."""
        rail = Rail.from_pydantic(
            output_class=output_class,
            prompt=prompt,
            instructions=instructions,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
        )
        return cast(
            Guard[Dict], cls(rail, num_reasks=num_reasks, base_model=output_class)
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
        rail = Rail.from_string_validators(
            validators=validators,
            description=description,
            prompt=prompt,
            instructions=instructions,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
        )
        return cast(Guard[str], cls(rail, num_reasks=num_reasks))

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
    ) -> Union[ValidationOutcome[OT], Iterable[str]]:
        ...

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
    ) -> Awaitable[ValidationOutcome[OT]]:
        ...

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
        Union[ValidationOutcome[OT], Iterable[str]], Awaitable[ValidationOutcome[OT]]
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
        if metadata is None:
            metadata = {}
        if full_schema_reask is None:
            full_schema_reask = self.base_model is not None
        if prompt_params is None:
            prompt_params = {}

        context = contextvars.ContextVar("kwargs")
        context.set(kwargs)

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

        # If the LLM API is async, return a coroutine
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
    ) -> Union[ValidationOutcome[OT], Iterable[str]]:
        instructions_obj = instructions or self.instructions
        prompt_obj = prompt or self.prompt
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
            with start_action(action_type="guard_call", prompt_params=prompt_params):
                runner = StreamRunner(
                    instructions=instructions_obj,
                    prompt=prompt_obj,
                    msg_history=msg_history_obj,
                    api=get_llm_ask(llm_api, *args, **kwargs),
                    prompt_schema=self.prompt_schema,
                    instructions_schema=self.instructions_schema,
                    msg_history_schema=self.msg_history_schema,
                    output_schema=self.output_schema,
                    num_reasks=num_reasks,
                    metadata=metadata,
                    base_model=self.base_model,
                    full_schema_reask=full_schema_reask,
                )
                return runner(call_log=call_log, prompt_params=prompt_params)
        else:
            # Otherwise, use Runner
            with start_action(action_type="guard_call", prompt_params=prompt_params):
                runner = Runner(
                    instructions=instructions_obj,
                    prompt=prompt_obj,
                    msg_history=msg_history_obj,
                    api=get_llm_ask(llm_api, *args, **kwargs),
                    prompt_schema=self.prompt_schema,
                    instructions_schema=self.instructions_schema,
                    msg_history_schema=self.msg_history_schema,
                    output_schema=self.output_schema,
                    num_reasks=num_reasks,
                    metadata=metadata,
                    base_model=self.base_model,
                    full_schema_reask=full_schema_reask,
                )
                call = runner(call_log=call_log, prompt_params=prompt_params)
                return ValidationOutcome[OT].from_guard_history(call)

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
        instructions_obj = instructions or self.instructions
        prompt_obj = prompt or self.prompt
        msg_history_obj = msg_history or []
        if prompt_obj is None:
            if msg_history_obj is not None and not len(msg_history_obj):
                raise RuntimeError(
                    "You must provide a prompt if msg_history is empty. "
                    "Alternatively, you can provide a prompt in the RAIL spec."
                )
        with start_action(action_type="guard_call", prompt_params=prompt_params):
            runner = AsyncRunner(
                instructions=instructions_obj,
                prompt=prompt_obj,
                msg_history=msg_history_obj,
                api=get_async_llm_ask(llm_api, *args, **kwargs),
                prompt_schema=self.prompt_schema,
                instructions_schema=self.instructions_schema,
                msg_history_schema=self.msg_history_schema,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                metadata=metadata,
                base_model=self.base_model,
                full_schema_reask=full_schema_reask,
            )
            call = await runner.async_run(
                call_log=call_log, prompt_params=prompt_params
            )
            return ValidationOutcome[OT].from_guard_history(call)

    def __repr__(self):
        return f"Guard(RAIL={self.rail})"

    def __rich_repr__(self):
        yield "RAIL", self.rail

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
    ) -> ValidationOutcome[OT]:
        ...

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
    ) -> Awaitable[ValidationOutcome[OT]]:
        ...

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
    ) -> ValidationOutcome[OT]:
        ...

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
        final_num_reasks = (
            num_reasks if num_reasks is not None else 0 if llm_api is None else None
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

        context = contextvars.ContextVar("kwargs")
        context.set(kwargs)

        input_prompt = self.prompt._source if self.prompt else None
        input_instructions = self.instructions._source if self.instructions else None
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
        with start_action(action_type="guard_parse"):
            runner = Runner(
                instructions=kwargs.pop("instructions", None),
                prompt=kwargs.pop("prompt", None),
                msg_history=kwargs.pop("msg_history", None),
                api=get_llm_ask(llm_api, *args, **kwargs) if llm_api else None,
                prompt_schema=self.prompt_schema,
                instructions_schema=self.instructions_schema,
                msg_history_schema=self.msg_history_schema,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                metadata=metadata,
                output=llm_output,
                base_model=self.base_model,
                full_schema_reask=full_schema_reask,
            )
            call = runner(call_log=call_log, prompt_params=prompt_params)

            return ValidationOutcome[OT].from_guard_history(call)

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
        with start_action(action_type="guard_parse"):
            runner = AsyncRunner(
                instructions=kwargs.pop("instructions", None),
                prompt=kwargs.pop("prompt", None),
                msg_history=kwargs.pop("msg_history", None),
                api=get_async_llm_ask(llm_api, *args, **kwargs) if llm_api else None,
                prompt_schema=self.prompt_schema,
                instructions_schema=self.instructions_schema,
                msg_history_schema=self.msg_history_schema,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                metadata=metadata,
                output=llm_output,
                base_model=self.base_model,
                full_schema_reask=full_schema_reask,
            )
            call = await runner.async_run(
                call_log=call_log, prompt_params=prompt_params
            )

            return ValidationOutcome[OT].from_guard_history(call)

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
